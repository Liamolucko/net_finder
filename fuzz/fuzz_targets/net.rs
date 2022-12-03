#![no_main]
use libfuzzer_sys::fuzz_target;
use net_finder::Net;

fuzz_target!(|net: Net| {
    // can't write `mut net` in args...
    let mut net = net;
    let width = net.width();
    net.height();
    for row in net.rows() {
        assert_eq!(row.len(), width);
    }
    for row in net.rows_mut() {
        assert_eq!(row.len(), width);
    }
    net.shrink();
    net.to_string();
});
