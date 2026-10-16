/* STM32H7B3I-DK firmware for the byte-level news encoder (MIND-Edge-Recommender).
 * Clock as set by STM32CubeMX for this board (and as on ST's benchmark board, "@280/140MHz"): HSE 24 MHz ->
 * PLL1 (M 12, N 280, P 2) -> 280 MHz CPU, AHB/AXI 140 MHz, APB 70 MHz, flash 3 wait states, VOS0, direct
 * SMPS supply; instruction and data caches on. Data, stack and the activations in DTCM (0x20000000), as the
 * activations pool of ST's benchmark. USART1 (PA9/PA10, the ST-LINK virtual COM port), 115200 8N1.
 * User LED LD1 on PG11, WAKEUP button on PC13. */
#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h7xx_hal.h"

#define BOARD_NAME    "STM32H7B3I-DK"
#define LED_Pin       GPIO_PIN_11
#define LED_Port      GPIOG
#define BUTTON_Pin    GPIO_PIN_13
#define BUTTON_Port   GPIOC

extern UART_HandleTypeDef huart1;
extern CRC_HandleTypeDef hcrc;

void MX_USART1_UART_Init(void);
void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif /* MAIN_H */
