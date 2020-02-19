# stochastics-labs

This program simulates the following problem:

Shoe shine shop has two chairs, one for brushing (1) and another for polishing (2).
Customers arrive according to PP with rate λ, and enter only if first chair is empty.
Shoe-shiner takes exp(μ1) time for brushing and exp(μ2) time for polishing.

Implementation is split in two crates: `cli` and `shoeshine_shop`. The first one takes care of command line arguments parsing, 
the second one is the library for this problem. Arrival/polishing distributions can be easily changed for something other than 
exponential.


### Usage
    sim.exe [OPTIONS] -i <iterations> --lambda <lambda> --mu1 <mu1> --mu2 <mu2>

    FLAGS:

        -h, --help
                Prints help information

        -V, --version
                Prints version information


    OPTIONS:

        -i <iterations>
                millions of events to simulate

        -l, --lambda <lambda>
                rate of customer arrival

            --mu1 <mu1>
                rate of serving on the first chair

            --mu2 <mu2>
                rate of serving on the second chair

        -s <seed>
                explicitly set seed

        -t <tail>
                change log tail [default: 0]

