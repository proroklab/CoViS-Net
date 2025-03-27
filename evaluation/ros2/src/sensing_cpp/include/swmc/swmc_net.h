#include <cstdint>

void init_log();

/// Starts the SWMC network stack. Once per process. The returned value is a pointer to a rust object and must not be modified in any way and only used by the calls specified below.
///  The caller of run() promises to execute stop() when the stack is no longer needed.
void* run(const char*);

/// Sends a message to the network.
///  Parameters: pointer to SWMC network object obtained from run(), stream id (use 2 if not certain), pointer to data array, length of data.
///  Returns true if send has been submitted OK
///  Limitations: Never send more than 64000 bytes per frame. (to be addressed in future)
bool send(const void*, uint16_t, const uint8_t[], uint32_t);

/// Returns the number of waiting messages
///  Parameters: pointer to SWMC network object obtained from run()
uint32_t receive_waiting_message_count(const void*);

/// Receives a message from the network
///  Parameters: pointer to SWMC network object obtained from run(), buffer for SWMC to use to store data (recommend 65536 bytes), length of provided buffer
///  Returns the number of bytes written to the buffer. Will return 0 if there were no messages, and -1 if the buffer was not big enough to store the message.
///   Warning: If -1 is returned due to limited buffer space, a message will be lost in the process
int32_t receive(const void*, uint8_t[], uint32_t);

/// Registers a callback function with SWMC
///  Parameters: pointer to SWMC network object obtained from run(), a function that accepts a buffer and buffer_len.
/// !!!!!!!!WARNING!!!!!!!! - you MUST call dealloc_buffer() with the passed values once you've finished with the buffer.
///         FAILURE TO COMPLY WILL LEAK MEMORY
bool register_receive_callback(const void*, void (*f)(uint8_t[], uint32_t));
/// Accepts the buffer and buffer length passed to a callback, freeing the Rust-allocated memory.
void dealloc_buffer(uint8_t[], uint32_t);

/// Stops the SWMC network stack.
void stop(void*);

