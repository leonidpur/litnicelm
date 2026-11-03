// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

class BackendInterface {
public:
    virtual ~BackendInterface() {}
    virtual void forward(int *tokens, int batch_size) = 0;
    virtual void train_step(int *tokens, int batch_size) = 0;
};
