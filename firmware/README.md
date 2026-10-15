# Board firmware (NUCLEO-F401RE)

Firmware for measuring the 8-bit encoders on a physical NUCLEO-F401RE (STM32F401RE, Cortex-M4 at 84 MHz),
built on this laptop without STM32CubeMX:

| app | what it does | how it is used |
|---|---|---|
| `validation` | ST's validation application (ST Edge AI Core 4.0, `Middlewares/ST/AI/Validation`) with the network | `stedgeai validate --mode target -d serial:COM3:115200` sends inputs, runs them on the board, reads outputs and DWT cycles: `scripts/board_validate.py` |
| `energy` | the network in a loop on one embedded MIND title, and STOP mode, switched with the blue button B1 | an external meter on the board's supply; the difference between the phases is the MCU's current for the encoder |

## Build and flash

```
python firmware/fetch_deps.py                      # STM32F4 HAL, CMSIS device and core headers, pinned commits
python firmware/nucleo_f401re/build.py artifacts/stedgeai/models/unas_f401_c142_seed42_body_int8qdq.onnx --app validation --flash
PYTHONPATH=. python -m scripts.board_validate --model unas_f401_c142 --port COM3
```

Local tools (not in the repository): ST Edge AI Core 4.0 (`C:\ST\STEdgeAI\4.0`: code generator, network runtime
library, validation application), GNU Tools for STM32 14.3 of STM32CubeIDE 2.1.1, STM32CubeProgrammer CLI.
The network code is generated with `--c-api legacy` (ST's validation application for the newer `st-ai` API
needs a multi-network wrapper that ST Edge AI Core does not ship); the kernels come from the same runtime
library either way. Generated code, objects and ELF files go to `nucleo_f401re/build/` (not committed).

`Core/Inc/app_x-cube-ai.h`, `Core/Src/app_x-cube-ai.c` and `Core/Inc/bsp_ai.h` are written by hand after the
templates of X-CUBE-AI 10.2.0 (`CubeMX/templates/*.ftl`) for one network and the validation application.

Clock as set by STM32CubeMX for this board: HSI 16 MHz, PLL to 84 MHz, flash 2 wait states, ART prefetch,
instruction and data caches on; USART2 (ST-LINK virtual COM port) at 115200 8N1.
