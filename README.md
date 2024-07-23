# AiAi AI 2
Super Monkey Ball AI using NEAT
![SMB Logo](./docs/SMB_series_logo_transparent.png)

## Contents
1. [Project Description](#Project-Description)
    1. [Technologies Used](#Technologies-Used)
1. [Inspiration & Purpose](#Inspiration-And-Purpose)
1. [Results](#Results)
    1. [Hardware Specs](#Hardware-Specs)
1. [How To Use](#How-To-Use)
    1. [Dolphin Emulation Config](#dolphin-emulation-config)
    1. [Requirements](#requirements)
    1. [Hotkey Config](#hotkey-config)
    1. [Controller Config](#controller-config)
    1. [Save State Setup](#save-state-setup)
    1. [Launch AI Agent](#launch-ai-agent)
    1. [Customizing Options](#Customizing-Options)
        1. [Runtime Arguments](#runtime-arguments)
        1. [Network](#network)
    1. [Known Issues](#Known-Issues)

## Project Description
This repository contains the excessive work completed in an attempt to create an artificial intelligence agent in the game *Super Monkey Ball 2*. Check out [this video](https://www.youtube.com/watch?v=JCW4fcwFM8M) on the previous iteration of the project.

### Technologies Used
This project utilizes [NEAT](https://github.com/CodeReclaimers/neat-python). The game is expected to be played on the [Dolphin Emulator](https://dolphin-emu.org).

#### NEAT
NEAT (Neural Evolution of Augmenting Topologies) is the primary mechanism utilized in this projects. To facilitate the evolution, the `aiai_ai.py` script runs the `neat-python` library. After a genome is initialized, a screenshot is taken at each interval in the script. This screenshot is then analyzed by the generated model and an action is determined accordingly.

![all_states](./docs/all_states.png)

In the `GOAL` state, the genome's fitness is determined by how quickly it reached the Goal, based on the following equation: `30 + 1.25 * time_remaining` where the fitness of the `GOAL` state is between `[30, 105]`.

In the `TIME OVER` and `FALL OUT` states, the fitness is determined by how close the agent got to the goal before failing the level--the closer it got at any point, the higher the fitness score.

The highest fitness score during a given genome's runtime will be the score evaluated during population reproduction.

## Inspiration And Purpose
The design of this project was largely inspired by SethBling's [MarI/O](https://www.youtube.com/watch?v=qv6UVOQ0F44). The inspiration of starting this project came from my wife holding one [World Records](https://www.speedrun.com/smbbm) in *Super Monkey Ball: Banana Mania* speedruns (as of July 22nd, 2024). I simply wanted to feel better about my inadequacy in this series.

The purpose of this project is to create a Super Monkey Ball AI that can reasonably beat standard stages in the game.

## Results
TBD

### Hardware Specs
All results were conducted on the following specs

```
Windows 10
NVIDIA RTX 3060
AMD Ryzen 5 3600 6-Core Processor 3.95 GHz
16 GB RAM
```

With these specs, and default configurations, the AI is able to analyze the game a bit over 30 FPS, essentially a little more than every other frame of the game.

## How To Use
To utilize this project, be sure to have a working version of the [Dolphin Emulator](https://dolphin-emu.org) installed on your machine, along with a ROM of [Super Monkey Ball 2 for GameCube](https://en.wikipedia.org/wiki/Super_Monkey_Ball_2). The project can be downloaded onto your local machine using the following command:

```
git clone https://github.com/AlexTheM8/AiAi-AI.git
```

### Dolphin Emulation Config
For best results, the following settings should be configured for the Dolphin Emulator.

- In Graphics settings, General, aspect ratio should be set to `Auto` or `Force 3:4`
- In Graphics settings, General, `Auto-Adjust Window Size` should be checked
- In Graphics settings, Enhancements, Internal Resolution should be set to `Native 640x528`
- In Config, Interface, `Keep Window on Top` should be checked
- In Config, Interface, `Show On-Screen Display Messages` should be un-checked
- In Config, Interface, `Show Active Title in Window Title` should be checked
- In Config, Interface, `Pause on Focus Loss` should be un-checked

### Requirements
The code of this project was programmed using `Python 3.11.0` on a Windows 10 machine. It is currently unknown if this project is functional in any other Python version or OS.

Once downloaded, navigate into the project folder and execute the following command to install the Python dependency libraries:

```
pip install -r requirements.txt
```

### Hotkey Config
Prior to executing the program, the virtual gamepad needs to be initialized. To do so, the `controller.py` must be configured executed.

Pulling up the Dolphin hotkey controls, navigate to the `Save and Load State` tab (see below). 

![dolphin_hotkey_default](./docs/dolphin_hotkey_default.PNG)

Once there, execute the following command:

```
python controller.py -s load
```

This will set the controller to repeatedly press the virtual button designated to loading the game's state. Follow these steps to configure the `Load State` hotkey.
 
1. In the Dolphin hotkeys menu, at the `Device` dropdown, select the virtual gamepad (multiple devices may be listed, so repeat steps 2-4 with a different device until the correct device is found)
1. Click on the `Load State Slot N` (where N can be any number you choose. **Remember this number**) box
1. Wait for `Button 3` or `Triangle` to appear in the box
1. Repeat steps 2 & 3 a few times if it does not appear immediately
1. You should then see `Button 3` or `Triangle` flash bold multiple times
1. Terminate the `controller.py` process
1. **[OPTIONAL]** Save the configuration as a `Profile` in Dolphin to avoid the need to repeat this process in the future

![dolphin_hotkey_config](./docs/dolphin_hotkey_config.png)

### Controller Config
For the joystick controls, navigate to the Dolphin controller config menu.

![dolphin_controls_default](./docs/dolphin_controls_default.PNG)

To configure the joystick in Dolphin, follow these steps:

1. Execute the command `python controller.py -s up`
1. In the Dolphin controller config menu, select the `Device` corresponding to the virtual gamepad (as described in [Hotkey Config](#hotkey-config), the device is not universally labeled, so steps 3-5 may need to be repeated until device is found)
1. Under the `Control Stick` section, click the box next to `Up`
1. Wait for `Axis Y-` to appear in the box
1. Repeat steps 3 & 4 a few times if it does not appear immediately
1. You should then see `Axis Y-` flash bold multiple times along with a visual representation of the joystick moving in the visual above
1. Terminate the `controller.py` process
1. Repeats steps 1-7 for each of the directions `[down, left, right]`, replacing each instance of "up" with the corresponding direction (`Axis Y+` for down, `Axis X+` for left, `Axis X-` for right)
1. **[OPTIONAL]** Save the configuration as a `Profile` in Dolphin to avoid the need to repeat this process in the future
1. **[OPTIONAL]** To test the controller configuration, execute the command `python controller.py -s random`. The virtual joystick should be moving appropriately

![dolphin_controls_config](./docs/dolphin_controls_config.png)

### Save State Setup
The evolution process requires a save state to be accessible as this is how the agent resets the game state upon starting a new genome. Please refer to [Hotkey Config](#hotkey-config) before configuring the save state. 

1. In the *Super Monkey Ball 2* game, navigate to the preferred stage
1. Save the game state at the start of the stage (before any time passes) in the same `Save Slot` chosen in the [Hotkey Config](#hotkey-config) step (**Note**: for a more-precise save state, use the frame-advance TAS tools provided by Dolphin)
1. Feel free to use the original code in `controller.py` to test if the save state (and corresponding hotkey) are configured appropriately (see [Hotkey Config](#hotkey-config)).

### Launch AI Agent
Once all the previous steps are completed, launch the *Super Monkey Ball 2* ROM on Dolphin and execute the following command:

```
python aiai_ai.py
```

From there, you should see the program load up the save state and begin evolution.
 
 **IMPORTANT**: After starting evolution DO NOT MOVE THE EMULATION WINDOW. If you want to move the window, restart the program.

### Customizing Options
To customize behavior, there are some options and features available in the AiAi AI. These options include logging and stat-tracking as well as network customization options.

#### Runtime Arguments
To customize the amount of messages displayed when running, there are three logging options: `[FULL, PARTIAL, NONE]`. The `FULL` logging option (enabled by default) will display all available logs during evolution. The `PARTIAL` logging option will display the max fitness of the most-recently executed genome along with the default NEAT logs. The `NONE` logging option will only display the default NEAT logs.

```
python aiai_ai.py --logging full
                -l       partial
                         none
```

Along with the evolution logging, there is a stat-tracking document that saves the max fitness, mean fitness, and standard deviation of each generation (saved as a CSV file labeled `stats.csv` under the `stats/[stage]` directory). This file is enabled by default, but can be disabled as an execution parameter.

```
python aiai_ai.py --stats
                -s
```

To speed up evolution, there is a feature where a given genome will be terminated early if it is not moving for too long. This is enabled by default, but can be disabled at execution. **NOTE:** feature should be disabled for stages that may require the player to wait.

```
python aiai_ai.py --zero_kill
                -z
```

#### Network
As part of the NEAT library, the neural network can be customized in a number of ways. For a full description of all customization options, see the [NEAT documentation](https://neat-python.readthedocs.io/en/latest/index.html). Some of these options in the `config-feedforward` document will be highlighted here.

In the `[NEAT]` section, the `fitness_threshold` can be adjusted for the given stage where the fitness can be `105` at max. The `pop_size`, or "population size" can be customized, where this number will correspond to how many genomes are in each generation.

Under the `network parameters` category of the `[DefaultGenome]` section, the `num_inputs` will be dependent on your window configurations. Best way to know what the `num_inputs` should be set to is by running the AiAi AI program (see [Launch AI Agent](#launch-ai-agent)) and change the value to the received value, in case of execution failure. See the example below.

```
File "c:\neat-python\neat\nn\recurrent.py", line 27, in activate
    raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))
RuntimeError: Expected 1425 inputs, got 5967
```

In this example, `num_inputs` should be set to 5967

### Known Issues
Below is a list of known issues with the current project.
