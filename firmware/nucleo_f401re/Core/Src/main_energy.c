/* NUCLEO-F401RE energy test for the byte-level news encoder.
 *
 * Two phases, switched with the blue button B1:
 *   loop  the network runs back to back on one embedded title (copied into its input buffer before every
 *         run); LD2 off, no serial traffic;
 *   stop  the MCU sits in STOP mode (flash powered down, low-power regulator, SRAM kept).
 * The board's supply is measured outside (a USB power meter between the laptop and the board). The other
 * consumers of the board (ST-LINK, power LED, regulators) draw the same in both phases, so the difference
 * between the phases is the current of the MCU running the encoder, less its STOP current.
 * LD2 blinks once when the loop starts and twice before STOP. When a loop phase ends, the board prints the
 * number of runs, the phase length and the mean DWT cycles per run over USART2 (115200 8N1). At start it
 * checks the output for the embedded title against the laptop's (title_input.h) and prints the cycles of
 * one run with and without the input copy. */
#include <stdio.h>
#include <string.h>
#include "main.h"
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
  HAL_GPIO_EXTI_IRQHandler(B1_Pin);
}

void HAL_GPIO_EXTI_Callback(uint16_t pin)
{
  if (pin == B1_Pin)
    button = 1;
}

static void print(void)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)line, strlen(line), 1000);
}

static void blink(int n)
{
  for (int i = 0; i < n; i++) {
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    HAL_Delay(150);
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
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
  HAL_Init();
  SystemClock_Config();
  DBGMCU->CR = 0;                         /* no debug clocks in STOP, whatever a debugger left set */
  MX_GPIO_Init(1);
  MX_USART2_UART_Init();
  MX_CRC_Init();

  ai_error err = ai_network_create_and_init(&network, data_activations0, NULL);
  if (err.type != AI_ERROR_NONE)
    Error_Handler();
  ai_in = ai_network_inputs_get(network, NULL);
  ai_out = ai_network_outputs_get(network, NULL);

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
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
  snprintf(line, sizeof(line), "energy test: %s, %lu Hz; output check: %d of 384 values differ (max %d)\r\n",
           AI_NETWORK_MODEL_NAME, (unsigned long)HAL_RCC_GetHCLKFreq(), diff, maxd);
  print();
  snprintf(line, sizeof(line), "one run: input copy %lu cycles, network %lu cycles\r\n",
           (unsigned long)(c1 - c0), (unsigned long)(c2 - c1));
  print();
  snprintf(line, sizeof(line), "phase loop started; press B1 to switch to STOP and back\r\n");
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
    HAL_SuspendTick();
    HAL_PWREx_EnableFlashPowerDown();
    HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);
    /* woken by B1: back to the 84 MHz clock */
    SystemClock_Config();
    HAL_ResumeTick();
    snprintf(line, sizeof(line), "stop: woke up\r\n");
    print();
    HAL_Delay(300);                       /* debounce */
  }
}
