use pyo3::prelude::*;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::str::FromStr;

#[pyfunction]
pub fn ip_to_float(
    ip4_devisor: f64,
    ip6_devisor: f64,
    error_value: f64,
    ip_address: &str,
) -> PyResult<f64> {
    Ok(match Ipv4Addr::from_str(ip_address) {
        Ok(ipv4) => (u128::from(u32::from(ipv4)) as f64) / ip4_devisor,
        Err(_) => match Ipv6Addr::from_str(ip_address) {
            Ok(ipv6) => (u128::from(ipv6) as f64) / ip6_devisor,
            Err(_) => error_value,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::ip_to_float;

    #[test]
    fn test_ip_to_float() {
        let ip4_devisor = 1.0;
        let ip6_devisor = 1.0;
        let error_value = -1.0;

        let ipv4_address = "192.168.1.1";
        let ipv6_address = "2001:0db8:85a3:0000:0000:8a2e:0370:7334";
        let invalid_address = "invalid_ip";

        assert_eq!(
            ip_to_float(ip4_devisor, ip6_devisor, error_value, ipv4_address).unwrap(),
            3232235777.0
        );
        assert_eq!(
            ip_to_float(ip4_devisor, ip6_devisor, error_value, ipv6_address).unwrap(),
            42540766452641154071740215577757643572.0
        );
        assert_eq!(
            ip_to_float(ip4_devisor, ip6_devisor, error_value, invalid_address).unwrap(),
            error_value
        );
    }
}
