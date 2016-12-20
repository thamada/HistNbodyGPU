#!/usr/bin/perl

my $nloop = 16;
my $logbase = "log.GTX";

for(my $ni=2; $ni<131073; $ni = $ni*2){
    my $bin = "\$\(BIN\)";
    my $ifile = "gen-plum/data.inp.".sprintf("%06d",$ni);
    my $log = "\$LOG";
    my $cmd = "$bin $ifile $nloop >> $log";
    print "\t".$cmd ."\n";
}

1;
