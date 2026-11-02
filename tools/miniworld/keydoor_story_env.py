#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import utils
from pyglet.gl import GL_TEXTURE_2D, glColor3f, glDisable, glPopMatrix, glPushMatrix, glRotatef, glTranslatef

from miniworld.entity import Box, COLORS, Entity
from miniworld.miniworld import MiniWorldEnv
from miniworld.opengl import drawBox
from miniworld.params import DEFAULT_PARAMS


FAST_PARAMS = DEFAULT_PARAMS.no_random()
FAST_PARAMS.set("forward_step", 0.7)
FAST_PARAMS.set("forward_drift", 0.0)
FAST_PARAMS.set("turn_step", 45)

LANE_Z_CHOICES = (2.2, 3.0, 3.8)
LAYOUT_TEMPLATES = (
    {
        "template_id": "migration_doorkey_canonical",
        "difficulty": "easy",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 4.2,
        "door_half_width": 0.65,
        "route_variant": "doorkey_canonical",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_forward_steps": 8,
        "primary_gate_color": "yellow",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_medium",
        "difficulty": "medium",
        "task_family": "doorkey",
        "room_size": 6.2,
        "agent_start_x": 1.5,
        "key_x": 2.85,
        "goal_dx": 4.8,
        "door_half_width": 0.58,
        "route_variant": "doorkey_medium",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 1,
        "post_pickup_forward_steps": 4,
        "post_toggle_forward_steps": 7,
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 4.4, "z_offset": -1.25, "size": [0.75, 0.85, 0.75], "color": "grey"},
            {"x": 8.4, "z_offset": 1.2, "size": [0.7, 0.8, 0.7], "color": "green"},
        ),
    },
    {
        "template_id": "migration_doorkey_hard",
        "difficulty": "hard",
        "task_family": "doorkey",
        "room_size": 6.2,
        "agent_start_x": 1.5,
        "key_x": 2.9,
        "goal_dx": 5.4,
        "door_half_width": 0.58,
        "route_variant": "doorkey_hard",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 1,
        "post_pickup_forward_steps": 5,
        "post_toggle_forward_steps": 8,
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.35,
        "clutter_specs": (
            {"x": 4.6, "z_offset": 1.35, "size": [0.8, 0.85, 0.8], "color": "green"},
            {"x": 7.4, "z_offset": -1.25, "size": [0.75, 0.8, 0.75], "color": "purple"},
            {"x": 10.0, "z_offset": 1.4, "size": [0.8, 0.9, 0.8], "color": "grey"},
        ),
    },
    {
        "template_id": "migration_multiroom_canonical",
        "difficulty": "easy",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 3.8,
        "door_half_width": 0.65,
        "route_variant": "multiroom_canonical",
        "world_variant": "three_room_two_gates",
        "connector_room_length": 2.6,
        "final_room_length": 3.8,
        "goal_room_margin_x": 1.9,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_medium",
        "difficulty": "medium",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 4.2,
        "door_half_width": 0.58,
        "route_variant": "multiroom_medium",
        "world_variant": "three_room_two_gates",
        "connector_room_length": 3.2,
        "final_room_length": 4.6,
        "goal_room_margin_x": 2.2,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "gate_open_side": "upper",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 7.9, "z_offset": 1.3, "size": [0.75, 0.8, 0.75], "color": "purple"},
            {"x": 11.4, "z_offset": -1.35, "size": [0.8, 0.9, 0.8], "color": "grey"},
        ),
    },
    {
        "template_id": "migration_multiroom_hard",
        "difficulty": "hard",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 4.2,
        "door_half_width": 0.58,
        "route_variant": "multiroom_hard",
        "world_variant": "three_room_two_gates",
        "connector_room_length": 3.6,
        "final_room_length": 5.0,
        "goal_room_margin_x": 2.4,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "gate_open_side": "lower",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "toggle_distance_slack": 0.35,
        "gate_open_side": "upper",
        "clutter_specs": (
            {"x": 7.2, "z_offset": -1.35, "size": [0.75, 0.8, 0.75], "color": "purple"},
            {"x": 10.2, "z_offset": 1.35, "size": [0.8, 0.9, 0.8], "color": "grey"},
            {"x": 13.1, "z_offset": -1.45, "size": [0.78, 0.85, 0.78], "color": "green"},
        ),
    },
    {
        "template_id": "migration_redblue_canonical",
        "difficulty": "easy",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_canonical",
        "world_variant": "three_room_two_gates",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "goal_required": False,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_medium",
        "difficulty": "medium",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.58,
        "route_variant": "redblue_medium",
        "world_variant": "three_room_two_gates",
        "connector_room_length": 3.2,
        "final_room_length": 3.2,
        "goal_required": False,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "gate_open_side": "upper",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 7.8, "z_offset": 1.3, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 10.8, "z_offset": -1.35, "size": [0.78, 0.85, 0.78], "color": "green"},
        ),
    },
    {
        "template_id": "migration_redblue_hard",
        "difficulty": "hard",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.52,
        "route_variant": "redblue_hard",
        "world_variant": "three_room_two_gates",
        "connector_room_length": 3.6,
        "final_room_length": 3.8,
        "goal_required": False,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "gate_open_side": "lower",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "toggle_distance_slack": 0.35,
        "clutter_specs": (
            {"x": 7.0, "z_offset": -1.3, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 10.1, "z_offset": 1.35, "size": [0.78, 0.85, 0.78], "color": "green"},
            {"x": 12.2, "z_offset": -1.45, "size": [0.8, 0.88, 0.8], "color": "purple"},
        ),
    },
    {
        "template_id": "migration_doorkey_structural",
        "difficulty": "structural",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "doorkey_structural_branch_goal",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 10,
        "turn_direction": "right90",
        "post_turn_forward_steps": 6,
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 8.0, "z_offset": -1.2, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 12.2, "z": 8.4, "size": [0.78, 0.85, 0.78], "color": "green"},
        ),
    },
    {
        "template_id": "migration_multiroom_structural",
        "difficulty": "structural",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "multiroom_structural_branch_goal",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 8.0, "z_offset": -1.2, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 12.0, "z": 8.2, "size": [0.78, 0.85, 0.78], "color": "green"},
            {"x": 9.8, "z": 8.8, "size": [0.8, 0.88, 0.8], "color": "purple"},
        ),
    },
    {
        "template_id": "migration_doorkey_structural_v2",
        "difficulty": "structural_v2",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "door_half_width": 0.65,
        "route_variant": "doorkey_structural_double_turn",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "upper",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "script_actions": [
            "pickup",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "primary_gate_color": "yellow",
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 8.0, "z_offset": -1.2, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 11.5, "z": 8.9, "size": [0.78, 0.85, 0.78], "color": "green"},
            {"x": 15.2, "z": 7.3, "size": [0.82, 0.9, 0.82], "color": "purple"},
        ),
    },
    {
        "template_id": "migration_multiroom_structural_v2",
        "difficulty": "structural_v2",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "multiroom_structural_double_turn",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "upper",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 8.0, "z_offset": -1.2, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 11.6, "z": 8.9, "size": [0.78, 0.85, 0.78], "color": "green"},
            {"x": 15.1, "z": 7.25, "size": [0.82, 0.9, 0.82], "color": "purple"},
        ),
    },
    {
        "template_id": "migration_redblue_structural",
        "difficulty": "structural",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_structural_four_room",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 1.45,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 7.8, "z_offset": -1.2, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 10.8, "z": 4.8, "size": [0.78, 0.85, 0.78], "color": "green"},
        ),
    },
    {
        "template_id": "migration_redblue_structural_v2",
        "difficulty": "structural_v2",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_structural_double_turn",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "upper",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 7.8, "z_offset": -1.2, "size": [0.75, 0.82, 0.75], "color": "grey"},
            {"x": 11.6, "z": 8.9, "size": [0.78, 0.85, 0.78], "color": "green"},
            {"x": 14.9, "z": 7.3, "size": [0.82, 0.9, 0.82], "color": "purple"},
        ),
    },
    {
        "template_id": "migration_doorkey_highdisc_branch_lower",
        "difficulty": "highdisc_branch",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.8,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_branch_lower",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.8,
        "final_room_length": 3.4,
        "branch_side": "lower",
        "branch_room_depth": 4.4,
        "branch_opening_min_x_offset": 1.1,
        "branch_opening_max_x_offset": 2.7,
        "branch_goal_x_offset": 2.35,
        "branch_goal_depth_margin": 2.55,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 11,
        "turn_direction": "left90",
        "post_turn_forward_steps": 7,
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_highdisc_zigzag_lower",
        "difficulty": "highdisc_zigzag",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_zigzag_lower",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.8,
        "final_room_length": 3.4,
        "terminal_room_length": 3.8,
        "goal_room_margin_x": 1.45,
        "branch_side": "lower",
        "branch_room_depth": 4.4,
        "branch_opening_min_x_offset": 1.0,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_depth_margin": 2.15,
        "script_actions": [
            "pickup",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_branch_lower",
        "difficulty": "highdisc_branch",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 1.8,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_branch_lower",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.8,
        "final_room_length": 3.4,
        "branch_side": "lower",
        "branch_room_depth": 4.4,
        "branch_opening_min_x_offset": 1.1,
        "branch_opening_max_x_offset": 2.7,
        "branch_goal_x_offset": 2.35,
        "branch_goal_depth_margin": 2.55,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_gate2_after_turn",
        "difficulty": "highdisc_zigzag",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_gate2_after_turn",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.8,
        "final_room_length": 3.4,
        "terminal_room_length": 3.8,
        "goal_room_margin_x": 1.45,
        "branch_side": "lower",
        "branch_room_depth": 4.4,
        "branch_opening_min_x_offset": 1.0,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_depth_margin": 2.15,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_highdisc_zigzag_upper",
        "difficulty": "highdisc_zigzag",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_highdisc_zigzag_upper",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.8,
        "final_room_length": 3.4,
        "terminal_room_length": 3.8,
        "goal_room_margin_x": 1.45,
        "branch_side": "upper",
        "branch_room_depth": 4.4,
        "branch_opening_min_x_offset": 1.0,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_depth_margin": 2.15,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_highdisc_zigzag_lower",
        "difficulty": "highdisc_zigzag",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_highdisc_zigzag_lower",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.8,
        "final_room_length": 3.4,
        "terminal_room_length": 3.8,
        "goal_room_margin_x": 1.45,
        "branch_side": "lower",
        "branch_room_depth": 4.4,
        "branch_opening_min_x_offset": 1.0,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_depth_margin": 2.15,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_highdisc_clear_zigzag",
        "difficulty": "highdisc_clear",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_clear_zigzag",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "upper",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "script_actions": [
            "pickup",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "primary_gate_color": "yellow",
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_highdisc_clear_zigzag_lower",
        "difficulty": "highdisc_clear",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_clear_zigzag_lower",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "lower",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "script_actions": [
            "pickup",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_clear_gate2_after_turn",
        "difficulty": "highdisc_clear",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_clear_gate2_after_turn",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "upper",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_clear_gate2_after_turn_lower",
        "difficulty": "highdisc_clear",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_clear_gate2_after_turn_lower",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "lower",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_highdisc_clear_order_zigzag",
        "difficulty": "highdisc_clear",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_highdisc_clear_order_zigzag",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "upper",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_highdisc_clear_order_zigzag_lower",
        "difficulty": "highdisc_clear",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_highdisc_clear_order_zigzag_lower",
        "world_variant": "five_stage_zigzag",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "terminal_room_length": 3.1,
        "goal_room_margin_x": 1.25,
        "branch_side": "lower",
        "branch_room_depth": 4.2,
        "branch_opening_min_x_offset": 0.95,
        "branch_opening_max_x_offset": 2.45,
        "branch_goal_depth_margin": 1.95,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_highdisc_clear_branch_upper",
        "difficulty": "highdisc_clear",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_clear_branch_upper",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 10,
        "turn_direction": "right90",
        "post_turn_forward_steps": 6,
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_highdisc_clear_branch_lower",
        "difficulty": "highdisc_clear",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_clear_branch_lower",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "lower",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 10,
        "turn_direction": "left90",
        "post_turn_forward_steps": 6,
        "primary_gate_color": "yellow",
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_doorkey_highdisc_clear_branch_lower_altopen",
        "difficulty": "highdisc_clear",
        "task_family": "doorkey",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "doorkey_highdisc_clear_branch_lower_altopen",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "lower",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 10,
        "turn_direction": "left90",
        "post_turn_forward_steps": 6,
        "primary_gate_color": "yellow",
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_clear_branch_upper",
        "difficulty": "highdisc_clear",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_clear_branch_upper",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_right",
            "turn_right",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_clear_branch_lower",
        "difficulty": "highdisc_clear",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_clear_branch_lower",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "lower",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_multiroom_highdisc_clear_branch_lower_altopen",
        "difficulty": "highdisc_clear",
        "task_family": "multiroom",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "multiroom_highdisc_clear_branch_lower_altopen",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "lower",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 2.2,
        "use_secondary_gate": True,
        "primary_gate_color": "yellow",
        "secondary_gate_color": "green",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "turn_left",
            "turn_left",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
        ],
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_highdisc_clear_four_room",
        "difficulty": "highdisc_clear",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_highdisc_clear_four_room",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 1.45,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "migration_redblue_highdisc_clear_four_room_lower",
        "difficulty": "highdisc_clear",
        "task_family": "redblue",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "door_half_width": 0.65,
        "route_variant": "redblue_highdisc_clear_four_room_lower",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.0,
        "branch_side": "lower",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.05,
        "branch_opening_max_x_offset": 2.55,
        "branch_goal_x_offset": 1.9,
        "branch_goal_depth_margin": 1.45,
        "goal_required": False,
        "use_secondary_gate": True,
        "primary_gate_color": "red",
        "secondary_gate_color": "blue",
        "script_actions": [
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "move_forward",
            "toggle",
        ],
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (),
    },
    {
        "template_id": "compact",
        "room_size": 5.8,
        "agent_start_x": 1.4,
        "key_x": 3.0,
        "goal_dx": 3.9,
        "door_half_width": 0.55,
        "route_variant": "straight",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 1,
        "post_pickup_forward_steps": 6,
        "post_toggle_forward_steps": 5,
        "clutter_specs": (),
    },
    {
        "template_id": "baseline",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 3.15,
        "goal_dx": 4.2,
        "door_half_width": 0.65,
        "route_variant": "straight",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 1,
        "post_pickup_forward_steps": 6,
        "post_toggle_forward_steps": 5,
        "clutter_specs": (),
    },
    {
        "template_id": "wide_door_far_goal",
        "room_size": 6.2,
        "agent_start_x": 1.6,
        "key_x": 3.25,
        "goal_dx": 4.5,
        "door_half_width": 0.8,
        "route_variant": "straight",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 1,
        "post_pickup_forward_steps": 6,
        "post_toggle_forward_steps": 5,
        "clutter_specs": (),
    },
    {
        "template_id": "narrow_door_mid_goal",
        "room_size": 6.1,
        "agent_start_x": 1.45,
        "key_x": 3.05,
        "goal_dx": 4.0,
        "door_half_width": 0.5,
        "route_variant": "straight",
        "world_variant": "two_room_straight",
        "pre_pickup_forward_steps": 1,
        "post_pickup_forward_steps": 6,
        "post_toggle_forward_steps": 5,
        "clutter_specs": (),
    },
    {
        "template_id": "three_stage_connector",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 4.2,
        "door_half_width": 0.65,
        "route_variant": "straight",
        "world_variant": "three_stage_connector",
        "connector_room_length": 2.6,
        "final_room_length": 4.6,
        "goal_room_margin_x": 2.1,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_forward_steps": 11,
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 4.1, "z_offset": -1.45, "size": [0.8, 1.0, 0.8], "color": "purple"},
            {"x": 8.2, "z_offset": 1.3, "size": [0.7, 0.8, 0.7], "color": "green"},
            {"x": 11.2, "z_offset": -1.5, "size": [0.8, 0.9, 0.8], "color": "grey"},
        ),
    },
    {
        "template_id": "three_stage_cluttered",
        "room_size": 6.15,
        "agent_start_x": 1.55,
        "key_x": 2.8,
        "goal_dx": 4.4,
        "door_half_width": 0.55,
        "route_variant": "straight",
        "world_variant": "three_stage_connector",
        "connector_room_length": 2.9,
        "final_room_length": 4.8,
        "goal_room_margin_x": 2.0,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_forward_steps": 11,
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 3.9, "z_offset": 1.55, "size": [0.9, 0.9, 0.9], "color": "green"},
            {"x": 7.9, "z_offset": -1.35, "size": [0.7, 0.85, 0.7], "color": "purple"},
            {"x": 10.1, "z_offset": 1.45, "size": [0.75, 0.8, 0.75], "color": "yellow"},
            {"x": 12.2, "z_offset": -1.55, "size": [0.8, 0.95, 0.8], "color": "grey"},
        ),
    },
    {
        "template_id": "three_stage_right_turn_goal",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "goal_z_offset": 1.4,
        "door_half_width": 0.65,
        "route_variant": "turn_right_goal",
        "world_variant": "three_stage_connector",
        "connector_room_length": 2.6,
        "final_room_length": 4.6,
        "goal_room_margin_x": 6.6,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 4,
        "turn_direction": "right90",
        "post_turn_forward_steps": 2,
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 4.1, "z_offset": -1.45, "size": [0.8, 1.0, 0.8], "color": "purple"},
            {"x": 8.3, "z_offset": -1.25, "size": [0.7, 0.8, 0.7], "color": "green"},
            {"x": 10.4, "z_offset": 1.65, "size": [0.8, 0.9, 0.8], "color": "grey"},
        ),
    },
    {
        "template_id": "three_stage_left_turn_goal",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "goal_z_offset": -1.4,
        "door_half_width": 0.65,
        "route_variant": "turn_left_goal",
        "world_variant": "three_stage_connector",
        "connector_room_length": 2.6,
        "final_room_length": 4.6,
        "goal_room_margin_x": 6.6,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 4,
        "turn_direction": "left90",
        "post_turn_forward_steps": 2,
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 4.1, "z_offset": 1.45, "size": [0.8, 1.0, 0.8], "color": "purple"},
            {"x": 8.3, "z_offset": 1.25, "size": [0.7, 0.8, 0.7], "color": "green"},
            {"x": 10.4, "z_offset": -1.65, "size": [0.8, 0.9, 0.8], "color": "grey"},
        ),
    },
    {
        "template_id": "four_stage_right_branch_room",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "turn_right_branch_room",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.2,
        "branch_side": "upper",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.15,
        "branch_opening_max_x_offset": 2.75,
        "branch_goal_x_offset": 2.05,
        "branch_goal_depth_margin": 1.6,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 10,
        "turn_direction": "right90",
        "post_turn_forward_steps": 6,
        "gate_open_side": "lower",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 4.1, "z_offset": -1.45, "size": [0.8, 1.0, 0.8], "color": "purple"},
            {"x": 8.2, "z_offset": 1.25, "size": [0.7, 0.8, 0.7], "color": "green"},
            {"x": 10.2, "z": 4.6, "size": [0.75, 0.9, 0.75], "color": "grey"},
            {"x": 10.4, "z": 8.7, "size": [0.7, 0.8, 0.7], "color": "green"},
        ),
    },
    {
        "template_id": "four_stage_left_branch_room",
        "room_size": 6.0,
        "agent_start_x": 1.5,
        "key_x": 2.75,
        "goal_dx": 1.6,
        "door_half_width": 0.65,
        "route_variant": "turn_left_branch_room",
        "world_variant": "four_stage_side_room",
        "connector_room_length": 2.6,
        "final_room_length": 3.2,
        "branch_side": "lower",
        "branch_room_depth": 3.8,
        "branch_opening_min_x_offset": 1.15,
        "branch_opening_max_x_offset": 2.75,
        "branch_goal_x_offset": 2.05,
        "branch_goal_depth_margin": 1.6,
        "pre_pickup_forward_steps": 0,
        "post_pickup_forward_steps": 4,
        "post_toggle_straight_steps": 10,
        "turn_direction": "left90",
        "post_turn_forward_steps": 6,
        "gate_open_side": "upper",
        "toggle_distance_slack": 0.3,
        "clutter_specs": (
            {"x": 4.1, "z_offset": 1.45, "size": [0.8, 1.0, 0.8], "color": "purple"},
            {"x": 8.2, "z_offset": -1.25, "size": [0.7, 0.8, 0.7], "color": "green"},
            {"x": 10.2, "z": 1.4, "size": [0.75, 0.9, 0.75], "color": "grey"},
            {"x": 10.4, "z": -2.7, "size": [0.7, 0.8, 0.7], "color": "green"},
        ),
    },
)


class StaticBox(Box):
    """A visible blocker/goal marker that cannot be picked up."""

    @property
    def is_static(self) -> bool:
        return True

    def render(self):
        if getattr(self, "visible_override", True) is False:
            return
        super().render()


class DecorativeBox(StaticBox):
    """Visible scene dressing with a tunable collision radius."""

    def __init__(
        self,
        color: str,
        size: list[float] | tuple[float, float, float],
        *,
        collision_radius: float = -1.0,
    ):
        super().__init__(color=color, size=size)
        self.radius = float(collision_radius)


class GateLeaf(Box):
    """Door leaf that can visually rotate open while using a tunable collision radius."""

    def __init__(self, color: str = "yellow", size: list[float] | tuple[float, float, float] = (0.18, 2.25, 1.12)):
        super().__init__(color=color, size=size)
        self.closed_radius = float(self.radius)
        self.open_radius = 0.08

    @property
    def is_static(self) -> bool:
        return True

    def set_closed_pose(self, pos: np.ndarray, dir_rad: float) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.dir = float(dir_rad)
        self.radius = self.closed_radius

    def set_open_pose(self, pos: np.ndarray, dir_rad: float) -> None:
        self.pos = np.asarray(pos, dtype=float)
        self.dir = float(dir_rad)
        self.radius = self.open_radius

    def render(self):
        if getattr(self, "visible_override", True) is False:
            return
        sx, sy, sz = self.size
        color_vec = getattr(self, "color_vec", COLORS[self.color])
        hinge_margin = 0.10
        knob_center_y = sy * 0.50
        knob_center_z = sz * 0.24
        front_face_x = sx / 2
        back_face_x = -sx / 2

        glDisable(GL_TEXTURE_2D)
        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180 / math.pi), 0, 1, 0)

        glColor3f(*color_vec)
        drawBox(
            x_min=-sx / 2,
            x_max=+sx / 2,
            y_min=0,
            y_max=sy,
            z_min=-sz / 2,
            z_max=+sz / 2,
        )

        # Hinge strip on the left edge of the door leaf.
        glColor3f(*COLORS["grey"])
        drawBox(
            x_min=-sx / 2 - 0.01,
            x_max=-sx / 2 + 0.03,
            y_min=hinge_margin,
            y_max=sy - hinge_margin,
            z_min=-sz / 2 + 0.06,
            z_max=-sz / 2 + 0.12,
        )
        drawBox(
            x_min=-sx / 2 - 0.01,
            x_max=-sx / 2 + 0.03,
            y_min=hinge_margin,
            y_max=sy - hinge_margin,
            z_min=sz / 2 - 0.12,
            z_max=sz / 2 - 0.06,
        )

        # Double-sided grey round-ish door knobs.
        for face_x, direction in ((front_face_x, 1.0), (back_face_x, -1.0)):
            glColor3f(0.22, 0.22, 0.22)
            drawBox(
                x_min=face_x + direction * 0.005,
                x_max=face_x + direction * 0.045,
                y_min=knob_center_y - 0.08,
                y_max=knob_center_y + 0.08,
                z_min=knob_center_z - 0.08,
                z_max=knob_center_z + 0.08,
            )
            glColor3f(0.72, 0.72, 0.72)
            drawBox(
                x_min=face_x + direction * 0.025,
                x_max=face_x + direction * 0.085,
                y_min=knob_center_y - 0.12,
                y_max=knob_center_y + 0.12,
                z_min=knob_center_z - 0.12,
                z_max=knob_center_z + 0.12,
            )
            glColor3f(0.90, 0.90, 0.90)
            drawBox(
                x_min=face_x + direction * 0.045,
                x_max=face_x + direction * 0.110,
                y_min=knob_center_y - 0.07,
                y_max=knob_center_y + 0.07,
                z_min=knob_center_z - 0.07,
                z_max=knob_center_z + 0.07,
            )

        glPopMatrix()


class VisibleKey(Entity):
    """Stylized side-view key silhouette tuned for both human audit and QA stability."""

    def __init__(self, color: str = "blue"):
        super().__init__()
        self.color = color
        self.radius = 0.24
        self.height = 0.34

    def randomize(self, params, rng):
        self.color_vec = COLORS[self.color] + params.sample(rng, "obj_color_bias")
        self.color_vec = np.clip(self.color_vec, 0, 1)

    def render(self):
        if getattr(self, "visible_override", True) is False:
            return
        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.color_vec)

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180 / math.pi), 0, 1, 0)

        # Head block
        drawBox(
            x_min=-0.58,
            x_max=-0.28,
            y_min=0.04,
            y_max=0.30,
            z_min=-0.07,
            z_max=0.07,
        )
        # Shaft
        drawBox(
            x_min=-0.28,
            x_max=0.28,
            y_min=0.12,
            y_max=0.20,
            z_min=-0.05,
            z_max=0.05,
        )
        # Teeth
        drawBox(
            x_min=0.16,
            x_max=0.28,
            y_min=0.05,
            y_max=0.20,
            z_min=-0.05,
            z_max=0.05,
        )
        drawBox(
            x_min=0.02,
            x_max=0.12,
            y_min=0.08,
            y_max=0.20,
            z_min=-0.05,
            z_max=0.05,
        )

        glPopMatrix()


class KeyDoorStoryEnv(MiniWorldEnv, utils.EzPickle):
    """MiniWorld environment with a key, gate and goal marker.
    
    The agent picks up the key, toggles the nearby gate and reaches the goal."""

    def __init__(
        self,
        room_size: float = 6.0,
        force_lane_z: float | None = None,
        force_template_id: str | None = None,
        **kwargs: Any,
    ):
        self.room_size = room_size
        self.force_lane_z = force_lane_z
        self.force_template_id = force_template_id
        self.layout_template_id = "baseline"
        self.difficulty = "default"
        self.task_family = "doorkey"
        self.agent_start_x = 1.5
        self.key_x = 3.15
        self.goal_dx = 4.2
        self.door_half_width = 0.65
        self.route_variant = "straight"
        self.world_variant = "two_room_straight"
        self.script_actions: list[str] | None = None
        self.primary_gate_color = "yellow"
        self.secondary_gate_color = "blue"
        self.goal_required = True
        self.use_secondary_gate = False
        self.connector_room_length = 0.0
        self.final_room_length = 0.0
        self.goal_room_margin_x = 0.0
        self.terminal_room_length = 0.0
        self.goal_z_offset = 0.0
        self.pre_pickup_forward_steps = 1
        self.post_pickup_forward_steps = 6
        self.post_toggle_forward_steps = 5
        self.post_toggle_straight_steps = 5
        self.turn_direction = "none"
        self.post_turn_forward_steps = 0
        self.gate_open_side = "lower"
        self.branch_side = "none"
        self.branch_room_depth = 0.0
        self.branch_opening_min_x_offset = 0.0
        self.branch_opening_max_x_offset = 0.0
        self.branch_goal_x_offset = 0.0
        self.branch_goal_depth_margin = 0.0
        self.toggle_distance_slack = 0.0
        self.clutter_specs: tuple[dict[str, Any], ...] = ()
        self.key_dir = math.pi / 2
        self.gate_closed_dir = 0.0
        self.gate_open_dir = math.pi / 2
        self.key_collected = False
        self.gate_open = False
        self.gate2_open = False
        self.red_door_open = False
        self.blue_door_open = False
        self.blue_opened_before_red = False
        self.goal_reached = False
        self.fail_reason: str | None = None
        self.intervention_type: str | None = None
        self.key: VisibleKey | None = None
        self.goal: StaticBox | None = None
        self.gate: GateLeaf | None = None
        self.gate2: GateLeaf | None = None
        self._taskc_visual_base_colors: dict[str, np.ndarray] = {}

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=40,
            params=kwargs.pop("params", FAST_PARAMS),
            domain_rand=kwargs.pop("domain_rand", False),
            **kwargs,
        )
        utils.EzPickle.__init__(self, room_size, force_lane_z, force_template_id, **kwargs)

    def _gen_world(self) -> None:
        if self.force_template_id is not None:
            template = None
            for candidate in LAYOUT_TEMPLATES:
                if candidate["template_id"] == self.force_template_id:
                    template = dict(candidate)
                    break
            if template is None:
                raise ValueError(f"Unknown force_template_id: {self.force_template_id}")
        else:
            template = dict(LAYOUT_TEMPLATES[self.np_random.choice(len(LAYOUT_TEMPLATES))])
        self.layout_template_id = str(template["template_id"])
        self.difficulty = str(template.get("difficulty", "default"))
        self.task_family = str(template.get("task_family", "doorkey"))
        self.room_size = float(template["room_size"])
        self.agent_start_x = float(template["agent_start_x"])
        self.key_x = float(template.get("key_x", self.key_x))
        self.goal_dx = float(template.get("goal_dx", self.goal_dx))
        self.door_half_width = float(template["door_half_width"])
        self.route_variant = str(template.get("route_variant", "straight"))
        self.world_variant = str(template.get("world_variant", "two_room_straight"))
        self.script_actions = list(template.get("script_actions", [])) or None
        self.primary_gate_color = str(template.get("primary_gate_color", "yellow"))
        self.secondary_gate_color = str(template.get("secondary_gate_color", "blue"))
        self.goal_required = bool(template.get("goal_required", True))
        self.use_secondary_gate = bool(
            template.get("use_secondary_gate", self.world_variant == "three_room_two_gates")
        )
        self.connector_room_length = float(template.get("connector_room_length", 0.0))
        self.final_room_length = float(template.get("final_room_length", 0.0))
        self.goal_room_margin_x = float(template.get("goal_room_margin_x", 0.0))
        self.terminal_room_length = float(template.get("terminal_room_length", 0.0))
        self.goal_z_offset = float(template.get("goal_z_offset", 0.0))
        self.pre_pickup_forward_steps = int(template.get("pre_pickup_forward_steps", 1))
        self.post_pickup_forward_steps = int(template.get("post_pickup_forward_steps", 6))
        self.post_toggle_forward_steps = int(template.get("post_toggle_forward_steps", 5))
        self.post_toggle_straight_steps = int(template.get("post_toggle_straight_steps", self.post_toggle_forward_steps))
        self.turn_direction = str(template.get("turn_direction", "none"))
        self.post_turn_forward_steps = int(template.get("post_turn_forward_steps", 0))
        self.gate_open_side = str(template.get("gate_open_side", "lower"))
        self.branch_side = str(template.get("branch_side", "none"))
        self.branch_room_depth = float(template.get("branch_room_depth", 0.0))
        self.branch_opening_min_x_offset = float(template.get("branch_opening_min_x_offset", 0.0))
        self.branch_opening_max_x_offset = float(template.get("branch_opening_max_x_offset", 0.0))
        self.branch_goal_x_offset = float(template.get("branch_goal_x_offset", 0.0))
        self.branch_goal_depth_margin = float(template.get("branch_goal_depth_margin", 0.0))
        self.toggle_distance_slack = float(template.get("toggle_distance_slack", 0.0))
        self.clutter_specs = tuple(dict(spec) for spec in template.get("clutter_specs", ()))
        self.key_dir = math.pi / 2
        lane_z = (
            float(self.force_lane_z)
            if self.force_lane_z is not None
            else float(LANE_Z_CHOICES[self.np_random.choice(len(LANE_Z_CHOICES))])
        )
        self.lane_z = lane_z
        self.gate2 = None
        self.goal = None
        room_a = self.add_rect_room(
            min_x=0,
            max_x=self.room_size,
            min_z=0,
            max_z=self.room_size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        gate_x = self.room_size + 0.25
        goal_x = self.room_size + self.goal_dx
        gate2_x = None
        goal_z = lane_z + self.goal_z_offset
        if self.world_variant in {"three_stage_connector", "four_stage_side_room", "three_room_two_gates", "five_stage_zigzag"}:
            connector_min_x = self.room_size + 0.5
            connector_max_x = connector_min_x + self.connector_room_length
            final_min_x = connector_max_x + 0.5
            final_max_x = final_min_x + self.final_room_length
            room_b = self.add_rect_room(
                min_x=connector_min_x,
                max_x=connector_max_x,
                min_z=0,
                max_z=self.room_size,
                wall_tex="brick_wall",
                floor_tex="asphalt",
                no_ceiling=True,
            )
            room_c = self.add_rect_room(
                min_x=final_min_x,
                max_x=final_max_x,
                min_z=0,
                max_z=self.room_size,
                wall_tex="brick_wall",
                floor_tex="asphalt",
                no_ceiling=True,
            )
            self.connect_rooms(
                room_a,
                room_b,
                min_z=lane_z - self.door_half_width,
                max_z=lane_z + self.door_half_width,
            )
            self.connect_rooms(
                room_b,
                room_c,
                min_z=lane_z - 0.95,
                max_z=lane_z + 0.95,
            )
            gate2_x = connector_max_x + 0.25
            gate2_lane_z = lane_z
            if self.world_variant in {"four_stage_side_room", "five_stage_zigzag"}:
                if self.branch_side == "upper":
                    branch_min_z = self.room_size + 0.5
                    branch_max_z = branch_min_z + self.branch_room_depth
                else:
                    branch_max_z = -0.5
                    branch_min_z = branch_max_z - self.branch_room_depth
                room_d = self.add_rect_room(
                    min_x=final_min_x,
                    max_x=final_max_x,
                    min_z=branch_min_z,
                    max_z=branch_max_z,
                    wall_tex="brick_wall",
                    floor_tex="asphalt",
                    no_ceiling=True,
                )
                self.connect_rooms(
                    room_c,
                    room_d,
                    min_x=final_min_x + self.branch_opening_min_x_offset,
                    max_x=final_min_x + self.branch_opening_max_x_offset,
                )
                branch_lane_z = (
                    branch_min_z + self.branch_goal_depth_margin
                    if self.branch_side == "upper"
                    else branch_max_z - self.branch_goal_depth_margin
                )
                if self.world_variant == "five_stage_zigzag":
                    elbow_min_x = final_max_x + 0.5
                    elbow_max_x = elbow_min_x + self.terminal_room_length
                    room_e = self.add_rect_room(
                        min_x=elbow_min_x,
                        max_x=elbow_max_x,
                        min_z=branch_min_z,
                        max_z=branch_max_z,
                        wall_tex="brick_wall",
                        floor_tex="asphalt",
                        no_ceiling=True,
                    )
                    self.connect_rooms(
                        room_d,
                        room_e,
                        min_z=branch_lane_z - 0.95,
                        max_z=branch_lane_z + 0.95,
                    )
                    gate2_x = final_max_x + 0.25
                    gate2_lane_z = branch_lane_z
                    goal_x = elbow_max_x - self.goal_room_margin_x
                    goal_z = branch_lane_z
                else:
                    goal_x = final_min_x + self.branch_goal_x_offset
                    goal_z = branch_lane_z
            else:
                goal_x = final_max_x - self.goal_room_margin_x
                goal_z = lane_z + self.goal_z_offset
        else:
            room_b = self.add_rect_room(
                min_x=self.room_size + 0.5,
                max_x=2 * self.room_size + 0.5,
                min_z=0,
                max_z=self.room_size,
                wall_tex="brick_wall",
                floor_tex="asphalt",
                no_ceiling=True,
            )
            self.connect_rooms(
                room_a,
                room_b,
                min_z=lane_z - self.door_half_width,
                max_z=lane_z + self.door_half_width,
            )
            goal_z = lane_z + self.goal_z_offset

        agent_start_z = lane_z
        key_z = lane_z
        distractor_z = lane_z + 0.85

        if self.task_family == "doorkey":
            self.key = self.place_entity(
                VisibleKey(color="blue"),
                pos=np.array([self.key_x, 0.0, key_z], dtype=float),
                dir=self.key_dir,
            )
        else:
            self.key = None
        self.distractor = self.place_entity(
            StaticBox(color="green", size=[0.7, 0.5, 0.5]),
            pos=np.array([self.key_x + 0.9, 0.0, distractor_z], dtype=float),
            dir=0.0,
        )
        if self.goal_required:
            self.goal = self.place_entity(
                StaticBox(color="red", size=[0.8, 0.8, 0.8]),
                pos=np.array([goal_x, 0.0, goal_z], dtype=float),
                dir=0.0,
            )
        door_height = 2.25
        self.gate_frame_entities = []
        self.gate_closed_pos = np.array([gate_x, 0.0, lane_z], dtype=float)
        gate_open_z = (
            lane_z - self.door_half_width + 0.12
            if self.gate_open_side != "upper"
            else lane_z + self.door_half_width - 0.12
        )
        self.gate_open_pos = np.array([gate_x + 0.42, 0.0, gate_open_z], dtype=float)
        self.gate = self.place_entity(
            GateLeaf(color=self.primary_gate_color, size=[0.18, door_height, 2 * self.door_half_width + 0.06]),
            pos=self.gate_closed_pos.copy(),
            dir=self.gate_closed_dir,
        )
        self.gate.set_closed_pose(self.gate_closed_pos, self.gate_closed_dir)
        self.gate_frame_entities.extend(self._place_gate_frame(gate_x=gate_x, lane_z=lane_z, door_height=door_height))
        if gate2_x is not None and self.use_secondary_gate:
            self.gate2_closed_pos = np.array([gate2_x, 0.0, gate2_lane_z], dtype=float)
            self.gate2_open_pos = np.array([gate2_x + 0.42, 0.0, gate_open_z], dtype=float)
            gate2_open_z = (
                gate2_lane_z - self.door_half_width + 0.12
                if self.gate_open_side != "upper"
                else gate2_lane_z + self.door_half_width - 0.12
            )
            self.gate2_open_pos = np.array([gate2_x + 0.42, 0.0, gate2_open_z], dtype=float)
            self.gate2 = self.place_entity(
                GateLeaf(color=self.secondary_gate_color, size=[0.18, door_height, 2 * self.door_half_width + 0.06]),
                pos=self.gate2_closed_pos.copy(),
                dir=self.gate_closed_dir,
            )
            self.gate2.set_closed_pose(self.gate2_closed_pos, self.gate_closed_dir)
            self.gate_frame_entities.extend(self._place_gate_frame(gate_x=gate2_x, lane_z=gate2_lane_z, door_height=door_height))
        self.extra_clutter = []
        for spec in self.clutter_specs:
            z = float(spec.get("z", lane_z + float(spec.get("z_offset", 0.0))))
            entity = self.place_entity(
                StaticBox(
                    color=str(spec.get("color", "white")),
                    size=list(spec.get("size", [0.7, 0.8, 0.7])),
                ),
                pos=np.array([float(spec["x"]), 0.0, z], dtype=float),
                dir=0.0,
            )
            self.extra_clutter.append(entity)

        self.place_agent(pos=np.array([self.agent_start_x, 0.0, agent_start_z], dtype=float), dir=0.0)
        self.key_collected = False
        self.gate_open = False
        self.gate2_open = False
        self.red_door_open = False
        self.blue_door_open = False
        self.blue_opened_before_red = False
        self.goal_reached = False
        self.fail_reason = None
        self.intervention_type = None
        self._snapshot_taskc_visual_base_colors()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Override camera after MiniWorld resets/randomizes the agent.
        self.agent.cam_pitch = -18
        self.agent.cam_fov_y = 72
        self._snapshot_taskc_visual_base_colors()
        obs = self.render_obs()
        return obs, info

    def _snapshot_taskc_visual_base_colors(self) -> None:
        self._taskc_visual_base_colors = {}
        for name in ("key", "goal", "gate", "gate2"):
            entity = getattr(self, name, None)
            color_vec = getattr(entity, "color_vec", None)
            if entity is not None and color_vec is not None:
                self._taskc_visual_base_colors[name] = np.asarray(color_vec, dtype=float).copy()

    def _attenuate_color(
        self,
        base: np.ndarray,
        *,
        target: np.ndarray,
        alpha: float,
        gain: float,
    ) -> np.ndarray:
        mixed = (1.0 - alpha) * np.asarray(base, dtype=float) + alpha * np.asarray(target, dtype=float)
        return np.clip(mixed * gain, 0.0, 1.0)

    def clear_taskc_visual_override(self) -> None:
        for name, base in self._taskc_visual_base_colors.items():
            entity = getattr(self, name, None)
            if entity is not None:
                entity.color_vec = np.asarray(base, dtype=float).copy()
                entity.visible_override = True

    def apply_taskc_visual_override(self, override_tag: str) -> None:
        self.clear_taskc_visual_override()
        if override_tag == "hide-goal":
            if self.goal is not None:
                self.goal.visible_override = False
            return
        if override_tag == "hide-doors":
            for gate_name in ("gate", "gate2"):
                entity = getattr(self, gate_name, None)
                if entity is not None:
                    entity.visible_override = False
            return
        if override_tag == "hide-key":
            if self.key is not None:
                self.key.visible_override = False
            return
        if override_tag == "dim-goal":
            if self.goal is not None and "goal" in self._taskc_visual_base_colors:
                floor_like = np.asarray([0.52, 0.50, 0.48], dtype=float)
                self.goal.color_vec = self._attenuate_color(
                    self._taskc_visual_base_colors["goal"],
                    target=floor_like,
                    alpha=0.97,
                    gain=0.58,
                )
            return
        if override_tag == "neutralize-doors":
            for gate_name in ("gate", "gate2"):
                entity = getattr(self, gate_name, None)
                if entity is not None and gate_name in self._taskc_visual_base_colors:
                    wall_like = np.asarray([0.46, 0.34, 0.26], dtype=float)
                    entity.color_vec = self._attenuate_color(
                        self._taskc_visual_base_colors[gate_name],
                        target=wall_like,
                        alpha=0.98,
                        gain=0.72,
                    )
            return
        if override_tag == "dim-key":
            if self.key is not None and "key" in self._taskc_visual_base_colors:
                wall_like = np.asarray([0.48, 0.36, 0.28], dtype=float)
                self.key.color_vec = self._attenuate_color(
                    self._taskc_visual_base_colors["key"],
                    target=wall_like,
                    alpha=0.96,
                    gain=0.66,
                )
            return

    def _place_gate_frame(self, *, gate_x: float, lane_z: float, door_height: float) -> list[DecorativeBox]:
        gate_frame_x = gate_x - 0.06
        frame_post_height = door_height + 0.08
        frame_post_size = [0.16, frame_post_height, 0.16]
        frame_lintel_height = 0.16
        frame_lintel_size = [0.16, frame_lintel_height, 2 * self.door_half_width + 0.32]
        return [
            self.place_entity(
                DecorativeBox(color="grey", size=frame_post_size),
                pos=np.array([gate_frame_x, 0.0, lane_z - self.door_half_width - 0.10], dtype=float),
                dir=0.0,
            ),
            self.place_entity(
                DecorativeBox(color="grey", size=frame_post_size),
                pos=np.array([gate_frame_x, 0.0, lane_z + self.door_half_width + 0.10], dtype=float),
                dir=0.0,
            ),
            self.place_entity(
                DecorativeBox(color="grey", size=frame_lintel_size),
                pos=np.array([gate_frame_x, frame_post_height - frame_lintel_height, lane_z], dtype=float),
                dir=0.0,
            ),
        ]

    def get_story_event_specs(self) -> list[dict[str, Any]]:
        if self.task_family == "multiroom":
            return [
                {"key": "gate_open", "label": "toggle-door1", "coverage": True},
                {"key": "gate2_open", "label": "toggle-door2", "coverage": True},
                {"key": "goal_reached", "label": "goal", "coverage": False},
            ]
        if self.task_family == "redblue":
            return [
                {"key": "red_door_open", "label": "toggle-red", "coverage": True},
                {"key": "blue_door_open", "label": "toggle-blue", "coverage": True},
                {"key": "goal_reached", "label": "terminal", "coverage": False},
            ]
        return [
            {"key": "key_collected", "label": "pickup", "coverage": True},
            {"key": "gate_open", "label": "toggle", "coverage": True},
            {"key": "goal_reached", "label": "goal", "coverage": False},
        ]

    def get_default_cf_intervention(self) -> str:
        if self.task_family == "doorkey":
            return "remove_key"
        if self.task_family == "redblue":
            return "swap_redblue_order"
        return "lock_gate2_forever"

    def apply_intervention(self, intervention_type: str) -> None:
        self.intervention_type = intervention_type
        if intervention_type == "remove_key":
            if self.key is not None and self.key in self.entities:
                self.entities.remove(self.key)
            if self.key is not None and self.agent.carrying is self.key:
                self.agent.carrying = None
            self._render_static()
        elif intervention_type == "lock_gate_forever":
            setattr(self, "gate_locked_forever", True)
        elif intervention_type == "lock_gate2_forever":
            setattr(self, "gate2_locked_forever", True)
        elif intervention_type == "swap_redblue_order":
            if self.task_family != "redblue" or self.gate is None or self.gate2 is None:
                raise ValueError("swap_redblue_order only supports redblue with two gates")
            self.primary_gate_color, self.secondary_gate_color = self.secondary_gate_color, self.primary_gate_color
            self.gate.color = self.primary_gate_color
            self.gate2.color = self.secondary_gate_color
            # GateLeaf instances cache color_vec during randomize(); update it too so
            # the visual storyboard actually reflects the swapped door order.
            self.gate.color_vec = COLORS[self.gate.color]
            self.gate2.color_vec = COLORS[self.gate2.color]
            self._snapshot_taskc_visual_base_colors()
            self._render_static()
        else:
            raise ValueError(f"Unsupported intervention_type: {intervention_type}")

    def _consume_key(self) -> None:
        if self.key is None:
            return
        if self.agent.carrying is self.key:
            self.agent.carrying = None
        if self.key in self.entities:
            self.entities.remove(self.key)

    def _semantic_color_for_gate(self, gate_name: str) -> str:
        if gate_name == "gate2":
            return str(self.secondary_gate_color)
        return str(self.primary_gate_color)

    def _mark_redblue_gate_open(self, gate_name: str) -> None:
        color = self._semantic_color_for_gate(gate_name)
        if color == "red":
            self.red_door_open = True
        elif color == "blue":
            self.blue_door_open = True

    def _open_gate(self, gate_name: str) -> None:
        if gate_name == "gate2":
            if self.gate2 is None or self.gate2_open:
                return
            self.gate2_open = True
            self.gate2.set_open_pose(self.gate2_open_pos, self.gate_open_dir)
            if self.task_family == "redblue":
                self._mark_redblue_gate_open("gate2")
            self._render_static()
            return
        if self.gate_open:
            return
        self.gate_open = True
        self.gate.set_open_pose(self.gate_open_pos, self.gate_open_dir)
        if self.task_family == "doorkey":
            self._consume_key()
        elif self.task_family == "redblue":
            self._mark_redblue_gate_open("gate1")
        self._render_static()

    def _update_flags(self) -> None:
        carrying = getattr(self.agent, "carrying", None)
        if self.key is not None and carrying is self.key:
            self.key_collected = True
            self._consume_key()

    def _current_fail_reason(self) -> str | None:
        if self.goal_reached:
            return None
        if self.task_family == "redblue":
            if self.blue_opened_before_red:
                return "blue_before_red"
            if not self.red_door_open:
                return "red_not_opened"
            return "blue_not_opened"
        if self.task_family == "multiroom":
            if not self.gate_open:
                return "door1_not_opened"
            if not self.gate2_open:
                return "door2_not_opened"
            return "goal_not_reached"
        if not self.key_collected:
            return "miss_key"
        if not self.gate_open:
            if self.near(self.gate):
                return "blocked_at_gate"
            return "wrong_toggle"
        return "timeout_after_partial_progress"

    def get_story_state(self) -> dict[str, Any]:
        visible = set()
        try:
            visible = set(self.get_visible_ents())
        except Exception:
            visible = set()

        key_present = self.key is not None and self.key in self.entities
        gate_present = self.gate is not None and self.gate in self.entities
        gate2_present = self.gate2 is not None and self.gate2 in self.entities
        carrying = getattr(self.agent, "carrying", None)
        return {
            "agent_pos": [
                round(float(self.agent.pos[0]), 3),
                round(float(self.agent.pos[2]), 3),
            ],
            "agent_dir_deg": round(math.degrees(float(self.agent.dir)), 2),
            "carrying": type(carrying).__name__ if carrying is not None else "none",
            "key_collected": bool(self.key_collected),
            "gate_open": bool(self.gate_open),
            "gate2_open": bool(self.gate2_open),
            "red_door_open": bool(self.red_door_open),
            "blue_door_open": bool(self.blue_door_open),
            "goal_reached": bool(self.goal_reached),
            "key_visible": key_present and self.key in visible,
            "gate_visible": gate_present and self.gate in visible,
            "gate2_visible": gate2_present and self.gate2 in visible,
            "goal_visible": self.goal is not None and self.goal in visible,
            "fail_reason": self.fail_reason,
        }

    def get_layout_metadata(self) -> dict[str, Any]:
        return {
            "template_id": self.layout_template_id,
            "difficulty": self.difficulty,
            "task_family": self.task_family,
            "route_variant": self.route_variant,
            "world_variant": self.world_variant,
            "room_size": self.room_size,
            "lane_z": round(float(self.lane_z), 3),
            "agent_start": [round(float(self.agent.pos[0]), 3), round(float(self.agent.pos[2]), 3)],
            "door_half_width": round(float(self.door_half_width), 3),
            "connector_room_length": round(float(self.connector_room_length), 3),
            "final_room_length": round(float(self.final_room_length), 3),
            "terminal_room_length": round(float(self.terminal_room_length), 3),
            "toggle_distance_slack": round(float(self.toggle_distance_slack), 3),
            "key_pos": (
                [round(float(self.key.pos[0]), 3), round(float(self.key.pos[2]), 3)]
                if self.key is not None
                else None
            ),
            "distractor_pos": [round(float(self.distractor.pos[0]), 3), round(float(self.distractor.pos[2]), 3)],
            "gate_pos": [round(float(self.gate.pos[0]), 3), round(float(self.gate.pos[2]), 3)],
            "gate_closed_pos": [round(float(self.gate_closed_pos[0]), 3), round(float(self.gate_closed_pos[2]), 3)],
            "gate_open_pos": [round(float(self.gate_open_pos[0]), 3), round(float(self.gate_open_pos[2]), 3)],
            "gate2_pos": (
                [round(float(self.gate2.pos[0]), 3), round(float(self.gate2.pos[2]), 3)]
                if self.gate2 is not None
                else None
            ),
            "goal_pos": (
                [round(float(self.goal.pos[0]), 3), round(float(self.goal.pos[2]), 3)]
                if self.goal is not None
                else None
            ),
            "clutter_count": len(getattr(self, "extra_clutter", [])),
            "gate_frame_count": len(getattr(self, "gate_frame_entities", [])),
            "goal_z_offset": round(float(self.goal_z_offset), 3),
            "turn_direction": self.turn_direction,
            "post_toggle_straight_steps": self.post_toggle_straight_steps,
            "post_turn_forward_steps": self.post_turn_forward_steps,
            "gate_open_side": self.gate_open_side,
            "branch_side": self.branch_side,
            "branch_room_depth": round(float(self.branch_room_depth), 3),
            "primary_gate_color": self.primary_gate_color,
            "secondary_gate_color": self.secondary_gate_color,
            "use_secondary_gate": self.use_secondary_gate,
            "key_style": "stylized_side_key",
            "key_dir_deg": round(math.degrees(float(self.key_dir)), 2),
            "gate_open_style": "rotate_90_keep_visible",
        }

    def get_story_action_sequence(self) -> list[str]:
        if self.script_actions is not None:
            return list(self.script_actions)
        turn_actions: list[str] = []
        if self.turn_direction == "right90":
            turn_actions = ["turn_right", "turn_right"]
        elif self.turn_direction == "left90":
            turn_actions = ["turn_left", "turn_left"]

        return (
            ["move_forward"] * self.pre_pickup_forward_steps
            + ["pickup"]
            + ["move_forward"] * self.post_pickup_forward_steps
            + ["toggle"]
            + ["move_forward"] * self.post_toggle_straight_steps
            + turn_actions
            + ["move_forward"] * self.post_turn_forward_steps
        )

    def _can_toggle_gate(self, gate: GateLeaf) -> bool:
        dist = float(np.linalg.norm(gate.pos - self.agent.pos))
        threshold = gate.radius + self.agent.radius + 1.1 * self.max_forward_step + self.toggle_distance_slack
        return dist < threshold

    def _pick_toggle_target(self) -> str | None:
        candidates: list[tuple[float, str, GateLeaf]] = []
        if self.gate is not None and not self.gate_open:
            candidates.append((float(np.linalg.norm(self.gate.pos - self.agent.pos)), "gate1", self.gate))
        if self.gate2 is not None and not self.gate2_open:
            candidates.append((float(np.linalg.norm(self.gate2.pos - self.agent.pos)), "gate2", self.gate2))
        candidates.sort(key=lambda x: x[0])
        for _, gate_name, gate in candidates:
            if self._can_toggle_gate(gate):
                return gate_name
        return None

    def step(self, action):
        gate_locked_forever = bool(getattr(self, "gate_locked_forever", False))
        gate2_locked_forever = bool(getattr(self, "gate2_locked_forever", False))
        obs, reward, termination, truncation, info = super().step(action)

        self._update_flags()
        if action == self.actions.pickup and self.key_collected:
            obs = self.render_obs()

        if action == self.actions.toggle and not gate_locked_forever:
            target = self._pick_toggle_target()
            has_key = self.key_collected or getattr(self.agent, "carrying", None) is self.key
            if target == "gate1":
                if self.task_family == "doorkey" and not has_key:
                    pass
                else:
                    if self.task_family == "redblue" and self._semantic_color_for_gate("gate1") == "blue" and not self.red_door_open:
                        self.blue_opened_before_red = True
                    self._open_gate("gate1")
                    obs = self.render_obs()
            elif target == "gate2":
                if gate2_locked_forever:
                    pass
                else:
                    if self.task_family == "redblue" and self._semantic_color_for_gate("gate2") == "blue" and not self.red_door_open:
                        self.blue_opened_before_red = True
                    self._open_gate("gate2")
                    obs = self.render_obs()

        if self.goal is not None and self.goal_required and self.near(self.goal):
            if self.task_family == "doorkey" and not self.gate_open:
                pass
            elif self.task_family == "multiroom" and not (self.gate_open and self.gate2_open):
                pass
            else:
                self.goal_reached = True
                reward += self._reward()
                termination = True

        if self.task_family == "redblue" and self.blue_door_open and self.red_door_open and not self.blue_opened_before_red:
            self.goal_reached = True
            reward += self._reward()
            termination = True

        if truncation and not self.goal_reached:
            self.fail_reason = self._current_fail_reason() or "other"
        elif termination and not self.goal_reached:
            self.fail_reason = self._current_fail_reason() or "other"
        else:
            self.fail_reason = None

        info = dict(info)
        info.update(self.get_story_state())
        info["success"] = bool(self.goal_reached)
        info["intervention_type"] = self.intervention_type
        return obs, reward, termination, truncation, info
