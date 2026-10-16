/* Energy test for the byte-level news encoder (any board with the interface of board.h).
 *
 * Two phases, switched with the user button:
 *   loop  the network runs back to back on one embedded title (copied into its input buffer before every
 *         run); user LED off, no serial traffic;
 *   stop  the MCU sits in STOP mode (board_stop).
 * The board's supply is measured outside (a USB power meter between the laptop and the board). The other
 * consumers of the board draw the same in both phases, so the difference between the phases is the power of
 * the MCU running the encoder, less its STOP power.
 * The LED blinks once when the loop starts and twice before STOP. When a loop phase ends, the board prints
 * the number of runs, the phase length and the mean DWT cycles per run over the virtual COM port (115200 8N1).
 * At start it checks the output for the embedded title against the laptop's (title_input.h) and prints the
 * cycles of one run with and without the input copy. */
#include <stdio.h>
#include <string.h>
#include "board.h"
#include "network.h"
#include "network_data.h"
#include "title_input.h"

AI_ALIGNED(32) static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];
static ai_handle data_activations0[] = { pool0 };
static ai_handle network = AI_HANDLE_NULL;
static ai_buffer *ai_in;
static ai_buffer *ai_out;
static volatile int button = 0;
static char line[160];

void EXTI15_10_IRQHandler(void)
{
  HAL_GPIO_EXTI_IRQHandler(BUTTON_Pin);
}

void HAL_GPIO_EXTI_Callback(uint16_t pin)
{
  if (pin == BUTTON_Pin)
    button = 1;
}

static void print(void)
{
  HAL_UART_Transmit(board_vcp, (uint8_t *)line, strlen(line), 1000);
}

static void blink(int n)
{
  for (int i = 0; i < n; i++) {
    board_led(1);
    HAL_Delay(150);
    board_led(0);
    HAL_Delay(250);
  }
}

static void copy_input(void)
{
  memcpy(ai_in[0].data, title_x, sizeof(title_x));
  memcpy(ai_in[1].data, title_mask, sizeof(title_mask));
}

static void run_once(void)
{
  copy_input();
  if (ai_network_run(network, ai_in, ai_out) != 1)
    Error_Handler();
}

int main(void)
{
  board_init(1);
  DBGMCU->CR = 0;                         /* no debug clocks in STOP, whatever a debugger left set */

  ai_error err = ai_network_create_and_init(&network, data_activations0, NULL);
  if (err.type != AI_ERROR_NONE)
    Error_Handler();
  ai_in = ai_network_inputs_get(network, NULL);
  ai_out = ai_network_outputs_get(network, NULL);

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if (__CORTEX_M == 7U)
  DWT->LAR = 0xC5ACCE55;                  /* Cortex-M7: unlock the DWT registers */
#endif
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  /* check: the output for the embedded title against the laptop's ONNX Runtime output (8-bit values) */
  uint32_t c0 = DWT->CYCCNT;
  copy_input();
  uint32_t c1 = DWT->CYCCNT;
  if (ai_network_run(network, ai_in, ai_out) != 1)
    Error_Handler();
  uint32_t c2 = DWT->CYCCNT;
  const int8_t *o = (const int8_t *)ai_out[0].data;
  int diff = 0, maxd = 0;
  for (int i = 0; i < 384; i++) {
    int d = o[i] - title_expected[i];
    if (d) diff++;
    if (d < 0) d = -d;
    if (d > maxd) maxd = d;
  }
  snprintf(line, sizeof(line), "energy test: %s, %s, CPU %lu Hz; output check: %d of 384 values differ (max %d)\r\n",
           BOARD_NAME, AI_NETWORK_MODEL_NAME, (unsigned long)SystemCoreClock, diff, maxd);
  print();
  snprintf(line, sizeof(line), "one run: input copy %lu cycles, network %lu cycles\r\n",
           (unsigned long)(c1 - c0), (unsigned long)(c2 - c1));
  print();
  snprintf(line, sizeof(line), "phase loop started; press the user button to switch to STOP and back\r\n");
  print();
  HAL_Delay(20);                          /* let the last bytes leave */

  while (1) {
    /* loop phase */
    blink(1);
    button = 0;
    uint32_t n = 0;
    uint64_t cycles = 0;
    uint32_t t0 = HAL_GetTick();
    while (!button) {
      uint32_t s = DWT->CYCCNT;
      run_once();
      cycles += (uint32_t)(DWT->CYCCNT - s);
      n++;
    }
    uint32_t ms = HAL_GetTick() - t0;
    snprintf(line, sizeof(line), "loop: %lu runs in %lu ms, mean %lu cycles per run\r\n",
             (unsigned long)n, (unsigned long)ms, (unsigned long)(n ? cycles / n : 0));
    print();

    /* stop phase */
    blink(2);
    HAL_Delay(200);
    button = 0;
    board_stop();
    snprintf(line, sizeof(line), "stop: woke up\r\n");
    print();
    HAL_Delay(300);                       /* debounce */
  }
}
