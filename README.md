# Nomo

A Computer Vision and System Programming project that implements an alternative cursor control system using eye tracking technology.

## Overview

Nomo is an innovative cursor control system that replaces traditional mouse input with real-time eye tracking. By leveraging computer vision and machine learning techniques, the project detects pupil movement and translates it into cursor positioning on the monitor. The application is inspired by the efficiency and keyboard-centric design philosophy of NeoVim.

## Project Structure

- **Prototype**: Initial implementation in Python using OpenCV and MediaPipe
- **Product**: Optimized production version implemented in C++

The development follows an iterative approach, with the Python prototype serving as the foundation for the final C++ implementation.

## Objectives

The primary goal of this project is to develop a reliable and responsive eye-tracking cursor control system that offers the following capabilities:

- Real-time pupil center detection and tracking from both eyes
- Accurate gaze point estimation through normalized coordinate mapping
- Smooth cursor movement with noise reduction algorithms
- Configurable edit modes accessible through keyboard shortcuts for enhanced usability

## Technical Approach

The system operates through the following pipeline:

1. Extract pupil center coordinates (x, y) from both eyes
2. Extract eye corner coordinates (x, y) from both eyes
3. Normalize horizontal axis: Map pupil center relative to eye corners
4. Normalize vertical axis: Map pupil center relative to cheekbone (current focus)
   - Implement calibration mechanism for vertical normalization based on frame data
5. Apply Exponential Moving Average (EMA) filter for noise reduction
6. Apply Kalman filter for output stabilization
7. Project normalized coordinates onto screen using linear transformation model
8. Implement calibration controls and mode switching functionality
9. Complete MVP and transition to production-ready implementation

## Technology Stack

- **Primary Languages**: Python (prototype), C++ (production)
- **Libraries**: OpenCV, MediaPipe (FaceMesh)
- **Computer Vision**: Facial landmark detection and eye tracking

## Status

This project is currently in active development with focus on finalizing the vertical normalization calibration and filter optimization.