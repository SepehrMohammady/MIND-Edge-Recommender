/* ST's validation application for the byte-level news encoder.
 * The host drives it with `stedgeai validate --mode target -d serial:<COM>:115200`, which sends the inputs,
 * runs the network on the board and reads back the outputs and the per-layer timing (DWT cycle counter). */
#include "board.h"
#include "app_x-cube-ai.h"

int main(void)
{
  board_init(0);
  MX_X_CUBE_AI_Init();
  while (1)
  {
    MX_X_CUBE_AI_Process();
  }
}
