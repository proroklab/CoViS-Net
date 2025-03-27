#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "image_transport/image_transport.hpp"
#include "sensing_msgs/msg/encoded_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <iostream>
#include <filesystem>
#include <cstddef>
#include <type_traits>

namespace swmc
{
extern "C"
{
#include "swmc/swmc_net.h"
}
}

using std::placeholders::_1;

template <typename T>
void serialize(std::vector<unsigned char>& vec, const T& value)
{
    static_assert(std::is_trivially_copyable<T>::value, "Serialization requires trivially copyable types");

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&value);

    for (size_t i = 0; i < sizeof(T); ++i)
    {
        // Serialize in big-endian order
        vec.push_back(bytes[sizeof(T) - 1 - i]);
    }
}

class EncodeImg : public rclcpp::Node
{
public:
    EncodeImg()
        : Node("encode_img")
        , img_proc_sub_{}
        , img_crop_size_{}
        , model_version_{}
        , swmc_{nullptr}
        , msg_seq_{0}
    {
        declare_parameter("image_inp_crop", 224);
        declare_parameter("model_enc_file", "");
        declare_parameter("swmc_config_file", "");

        std::string model_file;
        std::string swmc_config_file;
        get_parameter("model_enc_file", model_file);
        get_parameter("image_inp_crop", img_crop_size_);
        get_parameter("swmc_config_file", swmc_config_file);

        auto pkg_path = std::filesystem::path(ament_index_cpp::get_package_share_directory("sensing_cpp"));
        auto model_path = pkg_path / model_file;
        auto model_stem = model_path.stem().string();
        size_t pos = model_stem.find("_");
        model_version_ = model_stem.substr(0, pos);

        RCLCPP_INFO(get_logger(), "Loading model version %s from %s", model_version_.c_str(), model_path.c_str());

        model_enc_ = torch::jit::load(model_path);
        model_enc_.eval();
        model_enc_.to(torch::kCUDA);

        img_proc_sub_ =
            image_transport::create_subscription(
                this,
                "image_proc",
                std::bind(&EncodeImg::img_callback, this, _1),
                "raw",
                rmw_qos_profile_sensor_data
            );

        if (!swmc_config_file.empty())
        {
            auto swmc_config_path = pkg_path / swmc_config_file;
            swmc_ = swmc::run(swmc_config_path.c_str());

            //swmc::init_log();
        }
        else
        {
            enc_publisher_ =
                create_publisher<sensing_msgs::msg::EncodedImage>(
                    "enc",
                    rclcpp::SensorDataQoS()
                );
        }

        RCLCPP_INFO(get_logger(), "Initialized");
    }

    ~EncodeImg()
    {
        if (swmc_)
        {
            swmc::stop(swmc_);
        }

        RCLCPP_INFO(get_logger(), "Destroyed");
    }

private:
    image_transport::Subscriber img_proc_sub_;

    rclcpp::Publisher<sensing_msgs::msg::EncodedImage>::SharedPtr enc_publisher_;
    void* swmc_;

    torch::jit::Module model_enc_;
    int img_crop_size_;
    char msg_seq_;
    std::string model_version_;

    void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        float dt_rx = (get_clock()->now() - msg->header.stamp).nanoseconds() / 1e9;

        auto img_proc = img_to_tensor(*msg);
        auto inp_cuda = img_proc.permute({2, 0, 1}).to(torch::kFloat16).to(torch::kCUDA);

        auto img_crop = crop_center_square_tensor(inp_cuda, img_crop_size_);

        auto out = model_enc_.forward({img_crop.unsqueeze(0) / 255.0});
        auto out_cpu = out.toTensor().squeeze(0).to(torch::kFloat16).to(torch::kCPU);
        sensing_msgs::msg::EncodedImage enc = tensor_to_msg(out_cpu);
        enc.img_stamp = msg->header.stamp;
        enc.stamp = get_clock()->now();
        enc.model_version = model_version_;

        float dt_proc = (get_clock()->now() - msg->header.stamp).nanoseconds() / 1e9;

        if (swmc_)
        {
            auto swmc_msg = msg_to_swmc(enc);
            swmc::send(swmc_, 2, swmc_msg.data(), swmc_msg.size());
            RCLCPP_INFO(get_logger(), "Processed img rx %f proc %f via swmc", dt_rx, dt_proc);
        }
        else
        {
            enc_publisher_->publish(std::move(enc));
            RCLCPP_INFO(get_logger(), "Processed img rx %f proc %f via ros2", dt_rx, dt_proc);
        }

        msg_seq_++;
    }

    sensing_msgs::msg::EncodedImage tensor_to_msg(torch::Tensor& t)
    {
        assert((t.dtype() == torch::kFloat16) || (t.dtype() == torch::kFloat32));
        assert(t.device().is_cpu());

        sensing_msgs::msg::EncodedImage enc;
        enc.patches = t.size(0);
        enc.features = t.size(1);
        enc.dtype = t.dtype() == torch::kFloat16 ? "float16" : "float32";

        auto t_flat = t.flatten().contiguous();
        auto data_start = static_cast<unsigned char*>(t_flat.data_ptr());
        auto el_size = torch::elementSize(torch::typeMetaToScalarType(t_flat.dtype()));
        size_t data_len = t_flat.numel() * el_size;
        enc.data = std::vector<unsigned char>(data_start, data_start + data_len);

        return enc;
    }

    std::vector<unsigned char> msg_to_swmc(sensing_msgs::msg::EncodedImage& msg)
    {
        auto ns = std::string(get_namespace());
        size_t data_size = sizeof(size_t) + ns.size()
                           + sizeof(msg_seq_) // seq
                           + sizeof(size_t) + msg.model_version.size() // Model version, strlen
                           + sizeof(size_t) + msg.dtype.size() // dtype, strlen
                           + sizeof(msg.stamp.sec) // timestamp
                           + sizeof(msg.stamp.nanosec) // timestamp
                           + sizeof(msg.img_stamp.sec) // img stamp
                           + sizeof(msg.img_stamp.nanosec) // img stamp
                           + sizeof(msg.patches) // enc patches
                           + sizeof(msg.features) // enc features
                           + sizeof(size_t) // payload size
                           + msg.data.size(); // payload
        std::vector<unsigned char> swmc_data;
        swmc_data.reserve(data_size);

        serialize(swmc_data, ns.length());
        swmc_data.insert(swmc_data.end(), ns.begin(), ns.end());
        serialize(swmc_data, msg_seq_);
        serialize(swmc_data, msg.model_version.length());
        swmc_data.insert(swmc_data.end(), msg.model_version.begin(), msg.model_version.end());
        serialize(swmc_data, msg.dtype.length());
        swmc_data.insert(swmc_data.end(), msg.dtype.begin(), msg.dtype.end());
        serialize(swmc_data, msg.stamp.sec);
        serialize(swmc_data, msg.stamp.nanosec);
        serialize(swmc_data, msg.img_stamp.sec);
        serialize(swmc_data, msg.img_stamp.nanosec);
        serialize(swmc_data, msg.patches);
        serialize(swmc_data, msg.features);
        serialize(swmc_data, msg.data.size());
        swmc_data.insert(swmc_data.end(), msg.data.begin(), msg.data.end());
        return swmc_data;
    }

    torch::Tensor crop_center_square_tensor(const torch::Tensor& in, int size)
    {
        int height = in.size(1);
        int width = in.size(2);
        int start_h = std::max(0, (height - size) / 2);
        int start_w = std::max(0, (width - size) / 2);
        return in.narrow(1, start_h, size).narrow(2, start_w, size);
    }

    torch::Tensor img_to_tensor(const sensor_msgs::msg::Image& source) const
    {
        int byte_depth = sensor_msgs::image_encodings::bitDepth(source.encoding) / 8;
        int num_channels = sensor_msgs::image_encodings::numChannels(source.encoding);

        if (source.step < source.width * byte_depth * num_channels)
        {
            std::stringstream ss;
            ss << "Image is wrongly formed: step < width * byte_depth * num_channels  or  " <<
               source.step << " != " <<
               source.width << " * " << byte_depth << " * " << num_channels;
            throw std::runtime_error(ss.str());
        }

        if (source.height * source.step != source.data.size())
        {
            std::stringstream ss;
            ss << "Image is wrongly formed: height * step != size  or  " << source.height << " * " <<
               source.step << " != " << source.data.size();
            throw std::runtime_error(ss.str());
        }

        auto options = torch::TensorOptions().dtype(torch::kUInt8); //.device(torch::kCUDA);
        return torch::from_blob(const_cast<unsigned char*>(&source.data[0]), {source.height, source.width, num_channels}, options);
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EncodeImg>());
    rclcpp::shutdown();
    return 0;
}
