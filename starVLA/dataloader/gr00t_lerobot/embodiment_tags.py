# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    ActionNet = "action_net"
    """
    The ActionNet dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The OxE Bridge dataset.
    """

    OXE_RT1 = "oxe_rt1"
    """
    The OxE RT-1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    AGIBOT_BETA = "agibot_beta"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    EGODEX = "egodex"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NVWA_F = "nvwa_f"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    ROBOTWIN = "robotwin"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = 'franka'
    """
    The Franka Emika Panda robot.
    """

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_BRIDGE.value: 18,
    EmbodimentTag.OXE_RT1.value: 19,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.FRANKA.value: 25,
    EmbodimentTag.ActionNet.value: 27,
    EmbodimentTag.AGIBOT_BETA.value: 28,
    EmbodimentTag.EGODEX.value: 29,
    EmbodimentTag.NVWA_F.value: 30,
    EmbodimentTag.ROBOTWIN.value: 32
}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "oxe_droid": EmbodimentTag.OXE_DROID,
    "oxe_bridge": EmbodimentTag.OXE_BRIDGE,
    "oxe_rt1": EmbodimentTag.OXE_RT1,
    "demo_sim_franka_delta_joints": EmbodimentTag.FRANKA,
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
    "egodex": EmbodimentTag.EGODEX,
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
    "fourier_gr1_arms_waist2": EmbodimentTag.ActionNet,
    "agibot_genie1": EmbodimentTag.AGIBOT_GENIE1,
    "agibot_beta": EmbodimentTag.AGIBOT_BETA,
    "nvwa_f": EmbodimentTag.NVWA_F,
    "robotwin": EmbodimentTag.ROBOTWIN
}
