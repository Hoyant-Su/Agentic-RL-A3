import ipaddress

def is_ipv4(ip_str: str) -> bool:
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False

def is_ipv6(ip_str: str) -> bool:
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False
