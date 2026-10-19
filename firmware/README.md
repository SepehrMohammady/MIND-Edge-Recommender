# Board firmware (NUCLEO-F401RE, STM32H7B3I-DK)

Firmware for measuring the 8-bit encoders on physical boards, built on this laptop without STM32CubeMX:

| app | what it does | how it is used |
|---|---|---|
| `validation` | ST's validation application (ST Edge AI Core 4.0, `Middlewares/ST/AI/Validation`) with the network | `stedgeai validate --mode target -d serial:<COM>:115200` sends inputs, runs them on the board, reads outputs and DWT cycles: `scripts/board_validate.py`, `scripts/board_sweep.py` |
| `energy` | the network in a loop on one embedded MIND title, and STOP mode, switched with the user button | an external meter on the board's supply; the difference between the phases is the MCU's share: `scripts/board_energy.py` |
| `bnn` | the one-bit ReActNet encoder with TensorFlow Lite Micro and the Larq Compute Engine ops (`common/Src/main_bnn.cc`): checks 20 titles against LCE's interpreter, times them, prints each operator's cycles, then the energy loop | `scripts/board_bnn.py` parses the serial log |

Layout: `common/` holds the applications (`main_validation.c`, `main_energy.c`, the hand-written
`app_x-cube-ai.c/h` after the X-CUBE-AI 10.2.0 templates for one network) and the board interface
`board.h`; each board directory holds `main.h`, `bsp_ai.h`, the HAL configuration, `board.c` (clock tree, VCP
UART, user button and LED, STOP mode), the MSP and interrupt files and the linker script.

## Build and flash

```
python firmware/fetch_deps.py                      # STM32F4/H7 HAL, CMSIS device and core headers, pinned commits
python firmware/build.py --board nucleo_f401re artifacts/stedgeai/models/unas_f401_c142_seed42_body_int8qdq.onnx --app validation --flash
PYTHONPATH=. python -m scripts.board_validate --board NUCLEO-F401RE --model unas_f401_c142
python firmware/build.py --board stm32h7b3i_dk artifacts/stedgeai/models/unas_h7_c140_seed42_body_int8qdq.onnx --app validation --optimization time --flash
PYTHONPATH=. python -m scripts.board_validate --board STM32H7B3I-DK --model unas_h7_c140 --optimization time
PYTHONPATH=. python -m scripts.board_sweep --board STM32H7B3I-DK
python firmware/build.py --board stm32h7b3i_dk artifacts/binary_lce/reactnet_64-5-384_seed42_lce.tflite --app bnn --flash
```

`--optimization` passes `-O` to `stedgeai generate` (default `balanced`); `board_validate` must get the same
value as the flashed firmware. `--mempool` passes a memory-pool file (`stm32h7b3i_dk/mempools/` describes the
pools of ST's benchmark board).

Local tools (not in the repository): ST Edge AI Core 4.0 (`C:\ST\STEdgeAI\4.0`: code generator, network runtime
library, validation application), GNU Tools for STM32 14.3 of STM32CubeIDE 2.1.1, STM32CubeProgrammer CLI.
The network code is generated with `--c-api legacy` (ST's validation application for the newer `st-ai` API
needs a multi-network wrapper that ST Edge AI Core does not ship); the kernels come from the same runtime
library either way. Generated code, objects and ELF files go to `<board>/build/` (not committed).

Clocks as set by STM32CubeMX for each board:
- NUCLEO-F401RE: HSI 16 MHz, PLL to 84 MHz, flash 2 wait states, ART prefetch and caches on; USART2 (VCP).
- STM32H7B3I-DK: HSE 24 MHz, PLL1 to 280 MHz CPU, AHB/AXI 140 MHz, flash 3 wait states, VOS0, direct SMPS;
  instruction and data caches on; data, stack and activations in DTCM; USART1 (VCP). MCU current jumper JP2.

When a network takes 8-bit inputs (ST's generator folds the input quantisation into the graph for some
encoders), the energy firmware's embedded title is quantised on the laptop with the input's scale and zero
point, as an 8-bit byte table would do on a device.

The one-bit app uses TensorFlow Lite Micro and the Larq Compute Engine ops for it as ported to this board by the
lab's NAS-BNN project (copied by `fetch_deps.py`, pinned by a hash of the source trees). One fix is applied to the
copy: the border path of the binary convolution skipped padded taps; it now counts them as LCE pads (zero padding
in our model), which makes the board's outputs equal those of LCE's interpreter on the laptop. The tensor arena
sits in the AXI SRAM (`.axisram` section of the H7 linker script). Builds run three compiler jobs at below-normal
priority (`FW_JOBS`), since other work shares the laptop.
