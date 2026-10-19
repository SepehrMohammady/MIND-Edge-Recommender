/* One-bit ReActNet encoder on the board: TensorFlow Lite Micro with the Larq Compute Engine ops (LceQuantize,
 * LceBconv2d), the file of scripts/lce_convert.py.
 *
 * 1. check: each of the test titles of bnn_titles.h is looked up in the byte table on the MCU (the caller's job
 *    for every encoder of this work), run, and its 384 outputs are printed as hexadecimal floats together with
 *    the DWT cycles of the lookup and of Invoke(); the largest difference to the outputs of LCE's interpreter on
 *    the laptop (same file, same titles) is printed per title.
 * 2. timing: every title run BENCH_REPEATS more times; mean, min and max cycles of Invoke().
 * 3. energy: as common/Src/main_energy.c, the encoder in a loop on the first title and STOP mode, switched with
 *    the user button; when a loop phase ends the board prints the runs, the phase length and the mean cycles.
 * Serial output on the ST-LINK virtual COM port, 115200 8N1. */
#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
#include "board.h"
}
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "lce_ops_micro.h"
#include "bnn_model.h"     /* g_bnn_model[], g_bnn_model_len */
#include "bnn_titles.h"    /* N_TITLES, SEQ_LEN, EMBED_DIM, OUT_DIM, byte_table[], title_ids[][], lce_out[][] */

#ifndef ARENA_BYTES
#define ARENA_BYTES (512 * 1024)
#endif
#define BENCH_REPEATS 10

alignas(64) static uint8_t arena[ARENA_BYTES] __attribute__((section(".axisram")));

/* per-operator DWT cycles, filled by the interpreter (NAS-BNN's hook in micro_interpreter_graph.cc) */
extern "C" {
uint32_t g_op_cycles[256];
const char *g_op_names[256];
}
static volatile int button = 0;
static char line[200];

extern "C" void EXTI15_10_IRQHandler(void)
{
  HAL_GPIO_EXTI_IRQHandler(BUTTON_Pin);
}

extern "C" void HAL_GPIO_EXTI_Callback(uint16_t pin)
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

static TfLiteTensor *in_x, *in_m, *out;

/* the byte-table lookup and the mask (NHWC: x[l * 64 + e] = table[id_l * 64 + e]); ids 0..256, 0 = padding */
static void lookup(const uint16_t *ids)
{
  float *x = in_x->data.f, *m = in_m->data.f;
  for (int l = 0; l < SEQ_LEN; l++) {
    memcpy(x + l * EMBED_DIM, byte_table + ids[l] * EMBED_DIM, EMBED_DIM * sizeof(float));
    m[l] = ids[l] ? 1.0f : 0.0f;
  }
}

int main(void)
{
  board_init(1);
  DBGMCU->CR = 0;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if (__CORTEX_M == 7U)
  DWT->LAR = 0xC5ACCE55;
#endif
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddPrelu();
  resolver.AddSum();
  resolver.AddMaximum();
  resolver.AddDiv();
  resolver.AddFullyConnected();
  lce_micro::RegisterLCECustomOps(resolver);
  const tflite::Model *model = tflite::GetModel(g_bnn_model);
  static tflite::MicroInterpreter interp(model, resolver, arena, ARENA_BYTES);
  if (interp.AllocateTensors() != kTfLiteOk) {
    snprintf(line, sizeof(line), "bnn: AllocateTensors failed\r\n");
    print();
    Error_Handler();
  }
  /* inputs by shape: embedded bytes (1, 1, 128, 64) and mask (1, 1, 128, 1) */
  in_x = interp.input(0);
  in_m = interp.input(1);
  if (in_x->dims->data[3] != EMBED_DIM) {
    TfLiteTensor *t = in_x; in_x = in_m; in_m = t;
  }
  out = interp.output(0);
  snprintf(line, sizeof(line), "bnn: %s, CPU %lu Hz, model %lu B, arena used %lu of %lu B, %d titles\r\n",
           BOARD_NAME, (unsigned long)SystemCoreClock, (unsigned long)g_bnn_model_len,
           (unsigned long)interp.arena_used_bytes(), (unsigned long)ARENA_BYTES, N_TITLES);
  print();

  /* 1. check */
  for (int t = 0; t < N_TITLES; t++) {
    uint32_t c0 = DWT->CYCCNT;
    lookup(title_ids[t]);
    uint32_t c1 = DWT->CYCCNT;
    if (interp.Invoke() != kTfLiteOk)
      Error_Handler();
    uint32_t c2 = DWT->CYCCNT;
    float maxd = 0.0f;
    for (int i = 0; i < OUT_DIM; i++) {
      float d = fabsf(out->data.f[i] - lce_out[t][i]);
      if (d > maxd) maxd = d;
    }
    snprintf(line, sizeof(line), "title %d lookup %lu cycles invoke %lu cycles maxdiff_e9 %lu\r\n", t,
             (unsigned long)(c1 - c0), (unsigned long)(c2 - c1), (unsigned long)(maxd * 1e9f));
    print();
    for (int i = 0; i < OUT_DIM; i += 16) {
      int n = snprintf(line, sizeof(line), "out %d %d", t, i);
      for (int j = 0; j < 16; j++) {
        uint32_t u;
        memcpy(&u, &out->data.f[i + j], 4);
        n += snprintf(line + n, sizeof(line) - n, " %08lx", (unsigned long)u);
      }
      snprintf(line + n, sizeof(line) - n, "\r\n");
      print();
    }
  }

  /* 2. timing, and the cycles of each operator */
  memset(g_op_cycles, 0, sizeof(g_op_cycles));
  uint64_t sum = 0;
  uint32_t mn = 0xffffffffu, mx = 0;
  for (int r = 0; r < BENCH_REPEATS; r++) {
    for (int t = 0; t < N_TITLES; t++) {
      lookup(title_ids[t]);
      uint32_t c0 = DWT->CYCCNT;
      interp.Invoke();
      uint32_t c = DWT->CYCCNT - c0;
      sum += c;
      if (c < mn) mn = c;
      if (c > mx) mx = c;
    }
  }
  snprintf(line, sizeof(line), "bench: %d runs, invoke mean %lu min %lu max %lu cycles\r\n", BENCH_REPEATS * N_TITLES,
           (unsigned long)(sum / (BENCH_REPEATS * N_TITLES)), (unsigned long)mn, (unsigned long)mx);
  print();
  for (int i = 0; i < 256; i++) {
    if (g_op_names[i]) {
      snprintf(line, sizeof(line), "op %d %s mean %lu cycles\r\n", i, g_op_names[i],
               (unsigned long)(g_op_cycles[i] / (BENCH_REPEATS * N_TITLES)));
      print();
    }
  }
  snprintf(line, sizeof(line), "phase loop started; press the user button to switch to STOP and back\r\n");
  print();
  HAL_Delay(20);

  /* 3. energy: loop and STOP */
  while (1) {
    blink(1);
    button = 0;
    uint32_t n = 0;
    uint64_t cycles = 0;
    uint32_t t0 = HAL_GetTick();
    while (!button) {
      uint32_t s = DWT->CYCCNT;
      lookup(title_ids[0]);
      interp.Invoke();
      cycles += (uint32_t)(DWT->CYCCNT - s);
      n++;
    }
    uint32_t ms = HAL_GetTick() - t0;
    snprintf(line, sizeof(line), "loop: %lu runs in %lu ms, mean %lu cycles per run\r\n",
             (unsigned long)n, (unsigned long)ms, (unsigned long)(n ? cycles / n : 0));
    print();
    blink(2);
    HAL_Delay(200);
    button = 0;
    board_stop();
    snprintf(line, sizeof(line), "stop: woke up\r\n");
    print();
    HAL_Delay(300);
  }
}
