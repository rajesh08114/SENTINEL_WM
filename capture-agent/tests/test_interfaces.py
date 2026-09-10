from sentinel_capture.interfaces import Interface, list_interfaces, to_dicts


def test_list_interfaces_returns_dataclasses():
    ifaces = list_interfaces()
    assert len(ifaces) >= 1
    assert all(isinstance(i, Interface) for i in ifaces)
    for i in ifaces:
        assert isinstance(i.name, str) and i.name
        assert isinstance(i.is_up, bool)
        assert isinstance(i.is_loopback, bool)


def test_a_loopback_is_flagged():
    assert any(i.is_loopback for i in list_interfaces())


def test_to_dicts_roundtrips_keys():
    d = to_dicts(list_interfaces())
    assert d and set(d[0]) == {
        "name", "description", "ipv4", "netmask", "mac", "is_up", "is_loopback"}
