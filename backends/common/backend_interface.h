#pragma once

class BackendInterface {
public:
    virtual ~BackendInterface() {}
    virtual void forward(int *tokens, int batch_size) = 0;
    virtual void train_step(int *tokens, int batch_size) = 0;
};
