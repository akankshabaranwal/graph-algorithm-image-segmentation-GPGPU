#ifndef FELZENSWALB_OPTIONS_H
#define FELZENSWALB_OPTIONS_H


class Options
{
public:
    int warmupIterations = 0;
    int benchmarkIterations = 1;
    std::string inFile = "empty";
    std::string outFile = "empty";
};


#endif //FELZENSWALB_OPTIONS_H
