#ifndef SWMCNETWORK_H
#define SWMCNETWORK_H

#include <cstdint>
#include <functional>

// Include the C library header if it's separate
namespace swmc {
  extern "C" {
    #include "swmc/swmc_net.h"
  }
}

class SwmcNetwork {
public:
    SwmcNetwork() : swmcNetworkPtr(nullptr) {
        //swmc::init_log();
    }

    ~SwmcNetwork() {
        if (swmcNetworkPtr) {
            stop();
        }
    }

    // Disallow copy and assignment
    SwmcNetwork(const SwmcNetwork&) = delete;
    SwmcNetwork& operator=(const SwmcNetwork&) = delete;

    bool run(const char* config) {
        if (swmcNetworkPtr) {
            return false; // Already running
        }
        swmcNetworkPtr = run(config);
        return swmcNetworkPtr != nullptr;
    }

    bool send(uint16_t streamId, const uint8_t data[], uint32_t length) {
        if (!swmcNetworkPtr || length > 64000) {
            return false;
        }
        return send(swmcNetworkPtr, streamId, data, length);
    }

    uint32_t receiveWaitingMessageCount() {
        if (!swmcNetworkPtr) {
            return 0;
        }
        return receive_waiting_message_count(swmcNetworkPtr);
    }

    int32_t receive(uint8_t buffer[], uint32_t bufferLen) {
        if (!swmcNetworkPtr) {
            return -1;
        }
        return receive(swmcNetworkPtr, buffer, bufferLen);
    }

    bool registerReceiveCallback(const std::function<void(uint8_t[], uint32_t)>& callback) {
        if (!swmcNetworkPtr) {
            return false;
        }
        userCallback = callback;
        return register_receive_callback(swmcNetworkPtr, [](uint8_t* buf, uint32_t len) {
            SwmcNetwork::callbackHelper(buf, len);
        });
    }

    void stop() {
        if (swmcNetworkPtr) {
            stop(swmcNetworkPtr);
            swmcNetworkPtr = nullptr;
        }
    }

private:
    void* swmcNetworkPtr;
    std::function<void(uint8_t[], uint32_t)> userCallback;

    static void callbackHelper(uint8_t* buffer, uint32_t bufferLen) {
        if (SwmcNetwork::userCallback) {
            SwmcNetwork::userCallback(buffer, bufferLen);
            dealloc_buffer(buffer, bufferLen);
        }
    }

    // Declarations of the C functions
    extern "C" {
        void init_log();
        void* run(const char*);
        bool send(const void*, uint16_t, const uint8_t[], uint32_t);
        uint32_t receive_waiting_message_count(const void*);
        int32_t receive(const void*, uint8_t[], uint32_t);
        bool register_receive_callback(const void*, void (*)(uint8_t[], uint32_t));
        void dealloc_buffer(uint8_t[], uint32_t);
        void stop(void*);
    }
};

#endif // SWMCNETWORK_H
