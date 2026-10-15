/* NUCLEO-F401RE: ST's validation application for the byte-level news encoder.
 * The host drives it with `stedgeai validate --mode target -d serial:<COM>:115200`, which sends the inputs,
 * runs the network on the board and reads back the outputs and the per-layer timing (DWT cycle counter). */
#include "main.h"
#include "app_x-cube-ai.h"

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init(0);
  MX_USART2_UART_Init();
  MX_CRC_Init();           /* the network runtime library needs the CRC unit clocked */
  MX_X_CUBE_AI_Init();
  while (1)
  {
    MX_X_CUBE_AI_Process();
  }
}
