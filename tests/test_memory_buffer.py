"""Deploy-side episode memory buffer: cadence and alignment with the training layout."""

from openwam.deploy.memory_buffer import EpisodeMemoryBuffer, memory_times_for


def test_buffer_matches_training_layout_on_chunk_boundaries():
    buf = EpisodeMemoryBuffer(video_stride=4, max_latents=40)
    for t in range(16 * 3 + 1):  # control steps 0..48
        buf.observe(("frame", t))
        if t % 16 == 0:
            snap = buf.snapshot()
            c = t // 16
            assert snap["memory_context_index"] == c
            assert len(snap["memory_video"]) == 4 * c + 1
            assert snap["memory_video"][0] == ("frame", 0)
            assert snap["memory_video"][-1] == ("frame", t)
            assert snap["memory_times"] == list(range(c + 1))


def test_buffer_truncates_like_the_reader():
    assert memory_times_for(3, 40) == [0, 1, 2, 3]
    assert memory_times_for(10, 4) == [0, 8, 9, 10]
    buf = EpisodeMemoryBuffer(video_stride=4, max_latents=4)
    for t in range(16 * 10 + 1):
        buf.observe(t)
    snap = buf.snapshot()
    assert snap["memory_context_index"] == 10
    assert snap["memory_times"] == [0, 8, 9, 10]
    assert len(snap["memory_video"]) == 41  # full contiguous history; slots selected after encoding


def test_buffer_off_boundary_uses_last_complete_chunk_and_reset():
    buf = EpisodeMemoryBuffer(video_stride=4, max_latents=40)
    for t in range(23):
        buf.observe(t)
    snap = buf.snapshot()  # 6 kept frames (0,4,..,20) -> c = 1, frames 0..16
    assert snap["memory_context_index"] == 1
    assert snap["memory_video"] == [0, 4, 8, 12, 16]
    buf.reset()
    buf.observe(0)
    snap = buf.snapshot()
    assert snap["memory_context_index"] == 0 and snap["memory_video"] == [0] and snap["memory_times"] == [0]
