/* NUCLEO-F401RE firmware for the byte-level news encoder (MIND-Edge-Recommender).
 * Clock: HSI 16 MHz -> PLL -> 84 MHz SYSCLK/HCLK, APB1 42 MHz, APB2 84 MHz, flash 2 wait states with
 * prefetch, instruction and data caches on (the setting of STM32CubeMX for this board).
 * USART2 (PA2/PA3, the ST-LINK virtual COM port), 115200 8N1. LD2 on PA5, user button B1 on PC13. */
#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

#define BOARD_NAME    "NUCLEO-F401RE"
#define LD2_Pin       GPIO_PIN_5
#define LD2_GPIO_Port GPIOA
#define BUTTON_Pin    GPIO_PIN_13
#define BUTTON_Port   GPIOC

extern UART_HandleTypeDef huart2;
extern CRC_HandleTypeDef hcrc;

void MX_USART2_UART_Init(void);
void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif /* MAIN_H */
