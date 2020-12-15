#ifndef FELZENSWALB_OPTIONS_H
#define FELZENSWALB_OPTIONS_H


class Options
{
public:
    int warmupIterations = 0;
    int benchmarkIterations = 1;
    std::string inFile = "data/beach.png";
    std::string outFile = "segmented.png";
};


#endif //FELZENSWALB_OPTIONS_H
