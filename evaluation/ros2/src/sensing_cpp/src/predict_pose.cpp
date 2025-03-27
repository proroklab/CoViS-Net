#include <rclcpp/rclcpp.hpp>
#include <rcpputils/endian.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensing_msgs/msg/encoded_image.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <iostream>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <filesystem>
#include <cassert>
#include <algorithm>
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
T deserialize(std::vector<unsigned char>& vec)
{
    static_assert(std::is_trivially_copyable<T>::value, "Deserialization requires trivially copyable types");

    if (sizeof(T) > vec.size())
    {
        throw std::out_of_range("Not enough bytes in vector to deserialize the requested type");
    }

    T value = 0;
    const size_t typeSize = sizeof(T);

    for (size_t i = 0; i < typeSize; ++i)
    {
        value |= static_cast<T>(vec[i]) << ((typeSize - 1 - i) * 8);
    }

    // Erase the consumed bytes from the vector
    vec.erase(vec.begin(), vec.begin() + typeSize);

    return value;
}

static void static_swmc_callback(uint8_t* data, uint32_t length);

class PredictPose : public rclcpp::Node
{
public:
    PredictPose()
        : Node("predict_pose")
        , swmc_{nullptr}
    {
        declare_parameter("model_msg_file", "");
        declare_parameter("model_post_file", "");
        declare_parameter("cam_namespace_other", "");
        declare_parameter("pose_topic_name", "rel_pos");
        declare_parameter("swmc_config_file", "");

        std::string model_msg_file;
        std::string model_post_file;
        std::string pose_topic_name;
        std::string swmc_config_file;
        get_parameter("model_msg_file", model_msg_file);
        get_parameter("model_post_file", model_post_file);
        get_parameter("cam_namespace_other", cam_namespace_other_);
        get_parameter("pose_topic_name", pose_topic_name);
        get_parameter("swmc_config_file", swmc_config_file);

        assert(!cam_namespace_other_.empty());
        assert(!(cam_namespace_other_.back() == '/'));

        auto pkg_path = std::filesystem::path(ament_index_cpp::get_package_share_directory("sensing_cpp"));

        auto model_msg_path = pkg_path / model_msg_file;
        model_version_ = model_version_from_path(model_msg_path);
        RCLCPP_INFO(get_logger(), "Loading msg model version %s from %s", model_version_.c_str(), model_msg_path.c_str());
        model_msg_ = torch::jit::load(model_msg_path);
        model_msg_.eval();
        model_msg_.to(torch::kCUDA);

        auto model_post_path = pkg_path / model_post_file;
        assert(model_version_ == model_version_from_path(model_post_path));
        RCLCPP_INFO(get_logger(), "Loading post model from %s", model_post_path.c_str());
        model_post_ = torch::jit::load(model_post_path);
        model_post_.eval();
        model_post_.to(torch::kCUDA);

        if (swmc_config_file.empty())
        {
            std::ostringstream cam_topic_other;
            cam_topic_other << cam_namespace_other_ << "/enc";

            enc_sub_self_ =
                create_subscription<sensing_msgs::msg::EncodedImage>(
                    "enc",
                    rclcpp::SensorDataQoS(),
                    std::bind(&PredictPose::enc_self_callback, this, _1)
                );

            enc_sub_other_ =
                create_subscription<sensing_msgs::msg::EncodedImage>(
                    cam_topic_other.str(),
                    rclcpp::SensorDataQoS(),
                    std::bind(&PredictPose::enc_other_callback, this, _1)
                );
        }
        else
        {
            auto swmc_config_path = pkg_path / swmc_config_file;
            swmc_ = swmc::run(swmc_config_path.c_str());

            //swmc::init_log();
            swmc::register_receive_callback(swmc_, static_swmc_callback);

            /*
            timer_swmc_ =
                rclcpp::create_timer(
                    this,
                    get_clock(),
                    rclcpp::Duration(0, (int) (1.f / 30.f) * 1e9),
                    std::bind(&PredictPose::timer_swmc_callback, this)
                );
            */
        }

        pose_publisher_ =
            create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
                pose_topic_name,
                rclcpp::SensorDataQoS()
            );

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(get_logger(), "Initialized");
    }

    ~PredictPose()
    {
        if (swmc_)
        {
            swmc::stop(swmc_);
        }

        RCLCPP_INFO(get_logger(), "Destroyed");
    }

    void swmc_callback(uint8_t* data, uint32_t length)
    {
        auto buffer = std::vector<unsigned char>(data, data + length);

        auto enc_msg = std::make_shared<sensing_msgs::msg::EncodedImage>();
        auto namespace_sender = swmc_to_msg(buffer, enc_msg);
        auto const_enc_msg = std::const_pointer_cast<const sensing_msgs::msg::EncodedImage>(enc_msg);

        if (namespace_sender == std::string(get_namespace()))
        {
            enc_self_callback(const_enc_msg);
        }
        else if (namespace_sender == cam_namespace_other_)
        {
            enc_other_callback(const_enc_msg);
        }
        else
        {
            RCLCPP_DEBUG(get_logger(), "Received message from unspecified namespace %s", namespace_sender.c_str());
        }

        swmc::dealloc_buffer(data, length);
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensing_msgs::msg::EncodedImage>::SharedPtr enc_sub_self_;
    rclcpp::Subscription<sensing_msgs::msg::EncodedImage>::SharedPtr enc_sub_other_;
    rclcpp::TimerBase::SharedPtr timer_swmc_;
    void* swmc_;

    torch::jit::Module model_msg_;
    torch::jit::Module model_post_;
    std::string model_version_;
    std::string cam_namespace_other_;

    torch::Tensor other_enc_;;

    std::string model_version_from_path(std::filesystem::path& model_path)
    {
        auto model_stem = model_path.stem().string();
        size_t pos = model_stem.find("_");
        return model_stem.substr(0, pos);
    }

    void timer_swmc_callback(void)
    {
        auto own_namespace = std::string(get_namespace());

        auto waiting_messages = swmc::receive_waiting_message_count(swmc_);

        for (unsigned int i = 0; i < waiting_messages; i++)
        {
            std::vector<unsigned char> buffer;
            buffer.resize(65536);
            unsigned int rx_bytes = swmc::receive(swmc_, buffer.data(), 65536);
            buffer.resize(rx_bytes);

            auto enc_msg = std::make_shared<sensing_msgs::msg::EncodedImage>();
            auto namespace_sender = swmc_to_msg(buffer, enc_msg);
            auto const_enc_msg = std::const_pointer_cast<const sensing_msgs::msg::EncodedImage>(enc_msg);

            if (namespace_sender == own_namespace)
            {
                enc_self_callback(const_enc_msg);
            }
            else if (namespace_sender == cam_namespace_other_)
            {
                enc_other_callback(const_enc_msg);
            }
            else
            {
                RCLCPP_DEBUG(get_logger(), "Received message from unspecified namespace %s", namespace_sender.c_str());
            }
        }
    }

    void enc_self_callback(const sensing_msgs::msg::EncodedImage::ConstSharedPtr& enc)
    {
        if (other_enc_.numel() == 0)
        {
            RCLCPP_INFO(get_logger(), "Wait for other enc...");
            return;
        }

        float dt_rx = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;
        torch::Tensor self_enc = msg_to_tensor(enc);

        auto out = model_msg_.forward({self_enc.to(torch::kCUDA), other_enc_});
        auto pred = model_post_.forward({out.toTensor()}).toTuple()->elements();
        auto pos = pred[0].toTensor().squeeze(0).to(torch::kCPU);
        auto pos_var = pred[1].toTensor().squeeze(0).to(torch::kCPU);
        auto rot = pred[2].toTensor().squeeze(0).to(torch::kCPU);
        auto rot_var = pred[3].toTensor().squeeze(0).to(torch::kCPU);

        geometry_msgs::msg::PoseWithCovarianceStamped msg;
        msg.header.stamp = get_clock()->now();
        msg.pose.pose.position.x = pos[0].item<double>();
        msg.pose.pose.position.y = pos[1].item<double>();
        msg.pose.pose.position.z = pos[2].item<double>();
        msg.pose.pose.orientation.x = rot[0].item<double>();
        msg.pose.pose.orientation.y = rot[1].item<double>();
        msg.pose.pose.orientation.z = rot[2].item<double>();
        msg.pose.pose.orientation.w = rot[3].item<double>();

        auto cov = torch::diag(torch::cat({pos_var, rot_var.repeat(3)}, 0));
        auto cov_flat = cov.flatten().to(torch::kFloat64).contiguous();
        auto data_start = cov_flat.data_ptr<double>();
        auto el_size = torch::elementSize(torch::typeMetaToScalarType(cov_flat.dtype()));
        assert(el_size == sizeof(double));
        size_t data_len = cov_flat.numel() * el_size;
        std::copy(data_start, data_start + data_len, msg.pose.covariance.begin());

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = get_namespace();
        t.child_frame_id = cam_namespace_other_;

        t.transform.translation.x = msg.pose.pose.position.x;
        t.transform.translation.y = msg.pose.pose.position.y;
        t.transform.translation.z = msg.pose.pose.position.z;
        t.transform.rotation.x = msg.pose.pose.orientation.x;
        t.transform.rotation.y = msg.pose.pose.orientation.y;
        t.transform.rotation.z = msg.pose.pose.orientation.z;
        t.transform.rotation.w = msg.pose.pose.orientation.w;

        float dt_proc = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;

        pose_publisher_->publish(std::move(msg));
        tf_broadcaster_->sendTransform(t);

        RCLCPP_INFO(get_logger(), "Processed pose dt rx %f proc %f", dt_rx, dt_proc);
    }

    void enc_other_callback(const sensing_msgs::msg::EncodedImage::ConstSharedPtr& enc)
    {
        float dt_rx = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;

        torch::Tensor t = msg_to_tensor(enc);
        other_enc_ = t.to(torch::kCUDA);

        float dt_proc = (get_clock()->now() - enc->img_stamp).nanoseconds() / 1e9;

        RCLCPP_INFO(get_logger(), "Received other dt rx %f proc %f", dt_rx, dt_proc);
    }

    std::string swmc_to_msg(std::vector<unsigned char> swmc_data, sensing_msgs::msg::EncodedImage::SharedPtr& enc_msg)
    {
        size_t ns_length = deserialize<size_t>(swmc_data);
        auto ns_sender = std::string(swmc_data.begin(), swmc_data.begin() + ns_length);
        swmc_data.erase(swmc_data.begin(), swmc_data.begin() + ns_length);

        char seq = deserialize<char>(swmc_data);

        size_t model_version_length = deserialize<size_t>(swmc_data);
        enc_msg->model_version = std::string(swmc_data.begin(), swmc_data.begin() + model_version_length);
        swmc_data.erase(swmc_data.begin(), swmc_data.begin() + model_version_length);

        size_t dtype_length = deserialize<size_t>(swmc_data);
        enc_msg->dtype = std::string(swmc_data.begin(), swmc_data.begin() + dtype_length);
        swmc_data.erase(swmc_data.begin(), swmc_data.begin() + dtype_length);

        enc_msg->stamp.sec = deserialize<int32_t>(swmc_data);
        enc_msg->stamp.nanosec = deserialize<uint32_t>(swmc_data);
        enc_msg->img_stamp.sec = deserialize<int32_t>(swmc_data);
        enc_msg->img_stamp.nanosec = deserialize<uint32_t>(swmc_data);

        enc_msg->patches = deserialize<uint16_t>(swmc_data);
        enc_msg->features = deserialize<uint16_t>(swmc_data);

        size_t data_len = deserialize<size_t>(swmc_data);
        enc_msg->data = std::vector<unsigned char>(swmc_data.begin(), swmc_data.begin() + data_len);

        RCLCPP_INFO(get_logger(), "Received seq %d from %s data len %d", seq, ns_sender.c_str(), static_cast<int>(data_len));

        return ns_sender;
    }

    torch::Tensor msg_to_tensor(const sensing_msgs::msg::EncodedImage::ConstSharedPtr& msg)
    {
        assert((msg->dtype == "float16") || (msg->dtype == "float32"));
        auto dtype = msg->dtype == "float16" ? torch::kFloat16 : torch::kFloat32;
        auto options = torch::TensorOptions().dtype(dtype);
        size_t expected_raw_size = torch::elementSize(dtype) * msg->patches * msg->features;
        assert(msg->data.size() == expected_raw_size); //, "Unexpected message data size");
        return torch::from_blob(const_cast<unsigned char*>(msg->data.data()), {1, msg->patches, msg->features}, options);
    }
};

static std::shared_ptr<PredictPose> pose_node = nullptr;
static void static_swmc_callback(uint8_t* data, uint32_t length)
{
    if (pose_node)
    {
        pose_node->swmc_callback(data, length);
    }
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    pose_node = std::make_shared<PredictPose>();
    rclcpp::spin(pose_node);
    rclcpp::shutdown();
    return 0;
}
