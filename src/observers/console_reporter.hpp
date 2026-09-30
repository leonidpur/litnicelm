#pragma once

#include "training_observer.hpp"

#include <iostream>
#include <string>

// Prints training components' messages to the console.
class ConsoleReporter final : public ITrainingObserver {
public:
  void on_message(const std::string &line) override {
    std::cout << line << "\n";
  }
};
