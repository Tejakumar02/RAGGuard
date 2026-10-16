# Vision Inspection Station

This document is part of the fictional Kestrel Motors Plant 4 operations handbook, written as sample data for RAGGuard.

## Purpose

The vision inspection station watches the tower assembly fixture with a fixed overhead camera. It checks that every assembly step happens in the correct order and that no step is skipped.

## Camera and lighting

The camera streams 1080p video over RTSP at 25 frames per second. The ring light must stay on during production because detection accuracy drops sharply in shadow.

## Tower lamp signals

A green tower lamp means the current cycle is passing. An amber tower lamp means a step is in progress or the station is waiting for the operator. A red tower lamp with a buzzer means a step was skipped or done out of order, and the operator must correct it before the next unit.

## Daily start-up check

At the start of every shift, the operator runs the golden sample check by placing the reference assembly in the fixture. If the station does not show green within 30 seconds, the operator must call the automation engineer and must not start production.

## Lens cleaning

The camera lens is cleaned with a dry microfiber cloth once per shift. Solvents must never be used on the lens.
