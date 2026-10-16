/* Board interface used by the common applications (main_validation.c, main_energy.c).
 * Each board directory implements it in Core/Src/board.c; its main.h includes the HAL and defines
 * BOARD_NAME, BUTTON_Pin and the VCP UART handle. */
#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include "main.h"

/* HAL, caches (Cortex-M7), clock tree, GPIO (user button as a falling-edge interrupt when asked), the
 * ST-LINK virtual COM port UART (115200 8N1) and the CRC unit that the network runtime library needs. */
void board_init(int button_interrupt);

/* user LED on or off */
void board_led(int on);

/* STOP mode until the user button; the clock tree is restored on wake-up */
void board_stop(void);

/* the UART of the ST-LINK virtual COM port */
extern UART_HandleTypeDef *board_vcp;

#endif /* BOARD_H */
