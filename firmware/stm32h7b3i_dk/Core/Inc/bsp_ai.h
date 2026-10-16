/* Platform file expected by ST's validation application (ai_device_adaptor.h), written by hand after the
 * bsp_ai_h.ftl template of X-CUBE-AI 10.2.0: HAL, the application header and the UART handle. */
#ifndef BSP_H
#define BSP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "stm32h7xx.h"
#include "app_x-cube-ai.h"
#include "constants_ai.h"

#define UartHandle huart1
#define MX_UARTx_Init MX_USART1_UART_Init
void MX_USART1_UART_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* BSP_H */
