/* Application header for ST's validation firmware with one network ("network", legacy C API), written by
 * hand after the app_x-cube-ai_h.ftl template of X-CUBE-AI 10.2.0 (validation application, one model). */
#ifndef __APP_X_CUBE_AI_H
#define __APP_X_CUBE_AI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

#define MIN_HEAP_SIZE  0x2000
#define MIN_STACK_SIZE 0x1000

#define AI_MNETWORK_IN_1_SIZE_BYTES  AI_NETWORK_IN_1_SIZE_BYTES
#define AI_MNETWORK_IN_2_SIZE_BYTES  AI_NETWORK_IN_2_SIZE_BYTES
#define AI_MNETWORK_IN_NUM           AI_NETWORK_IN_NUM
#define AI_MNETWORK_OUT_1_SIZE_BYTES AI_NETWORK_OUT_1_SIZE_BYTES
#define AI_MNETWORK_OUT_NUM          AI_NETWORK_OUT_NUM

/* IO buffers */
extern ai_i8* data_ins[];
extern ai_i8* data_outs[];
extern ai_handle data_activations0[];

void MX_X_CUBE_AI_Init(void);
void MX_X_CUBE_AI_Process(void);

/* Multiple network support */
typedef struct {
    const char *name;
    ai_buffer *config;
    ai_bool (*ai_data_params_get)(ai_network_params* params);
    ai_bool (*ai_get_report)(ai_handle network, ai_network_report* report);
    ai_error (*ai_create)(ai_handle* network, const ai_buffer* network_config);
    ai_error (*ai_get_error)(ai_handle network);
    ai_handle (*ai_destroy)(ai_handle network);
    ai_bool (*ai_init)(ai_handle network, const ai_network_params* params);
    ai_i32 (*ai_run)(ai_handle network, const ai_buffer* input, ai_buffer* output);
    ai_i32 (*ai_forward)(ai_handle network, const ai_buffer* input);
    ai_handle * activations;
} ai_network_entry_t;

#define AI_MNETWORK_NUMBER  (1)

AI_API_DECLARE_BEGIN

AI_API_ENTRY
const char* ai_mnetwork_find(const char *name, ai_int idx);

AI_API_ENTRY
ai_bool ai_mnetwork_get_report(ai_handle network, ai_network_report* report);

AI_API_ENTRY
ai_error ai_mnetwork_get_error(ai_handle network);

AI_API_ENTRY
ai_error ai_mnetwork_create(const char *name, ai_handle* network, const ai_buffer* network_config);

AI_API_ENTRY
ai_handle ai_mnetwork_destroy(ai_handle network);

AI_API_ENTRY
ai_bool ai_mnetwork_init(ai_handle network);

AI_API_ENTRY
ai_i32 ai_mnetwork_run(ai_handle network, const ai_buffer* input, ai_buffer* output);

AI_API_ENTRY
ai_i32 ai_mnetwork_forward(ai_handle network, const ai_buffer* input);

AI_API_ENTRY
int ai_mnetwork_get_private_handle(ai_handle network, ai_handle *phandle, ai_network_params* pparams);

AI_API_DECLARE_END

#ifdef __cplusplus
}
#endif

#endif /* __APP_X_CUBE_AI_H */
