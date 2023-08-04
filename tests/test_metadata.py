"""Testing module for api metadata.
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument


def test_authors(metadata):
    """Tests that metadata provides authors information."""
    assert 'authors' in metadata
    assert isinstance(metadata['authors'], str)


def test_description(metadata):
    """Tests that metadata provides description information."""
    assert 'description' in metadata
    assert isinstance(metadata['description'], str)


def test_license(metadata):
    """Tests that metadata provides license information."""
    assert 'license' in metadata
    assert isinstance(metadata['license'], str)


def test_version(metadata):
    """Tests that metadata provides version information."""
    assert 'version' in metadata
    assert isinstance(metadata['version'], str)
    assert all(v.isnumeric() for v in metadata['version'].split('.'))
    assert len(metadata['version'].split('.')) == 3


def test_checkpoints_local(metadata):
    """Tests that metadata provides ckpt local information."""
    assert 'checkpoints_local' in metadata
    assert isinstance(metadata['checkpoints_local'], list)


def test_checkpoints_remote(metadata):
    """Tests that metadata provides backbone remote information."""
    assert 'checkpoints_remote' in metadata
    assert isinstance(metadata['checkpoints_remote'], list)




