//
// Created by gyorgy on 13/12/2020.
//

#ifndef FELZENSWALB_OPTIONS_H
#define FELZENSWALB_OPTIONS_H


class Options
{
public:
    bool useCPU = false;
    bool show = false;
    int warmupIterations = 1;
    int benchmarkIterations = 10;
    float sigma = 1.0f;
    uint k = 200;
    std::string inFile = "data/beach.png";
    std::string outFile = "segmented.png";
};


#endif //FELZENSWALB_OPTIONS_H
