import bisect
from collections import deque
from typing import Callable, Optional

from geometry_msgs.msg import TransformStamped
from message_filters import SimpleFilter
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time as RosTime
from tf2_ros import ExtrapolationException  # pyright: ignore[reportAttributeAccessIssue]
from tf2_ros.buffer import Buffer

_QueueEntry = tuple[RosTime, RosTime, tuple[object, ...]]  # (stamp, enqueued_at, messages)


class TfSampleTrackingBuffer(Buffer):
    """tf2_ros.Buffer that also records the timestamps broadcast directly under ``root_frame``.

    A resolved lookup only reports the *queried* time, not the underlying sample's real
    timestamp, so this is needed to tell how far a (possibly interpolated) transform's nearest
    real broadcast lies from the time it was looked up for. Only transforms with
    ``header.frame_id == root_frame`` are tracked (e.g. the one edge a SLAM system publishes
    straight off the map frame), since deeper links - a camera's internal rig, mount plates -
    are often re-broadcast on ``/tf`` at high rate too and would otherwise mask a stale pose.
    """

    def __init__(
        self, root_frame: str, *args: object, retention: Duration = Duration(seconds=5.0), **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.root_frame = root_frame
        self.retention = retention
        self._sample_stamps: list[RosTime] = []

    def set_transform(self, transform: TransformStamped, authority: str) -> None:
        super().set_transform(transform, authority)
        if transform.header.frame_id != self.root_frame:
            return
        stamp = RosTime.from_msg(transform.header.stamp)
        bisect.insort(self._sample_stamps, stamp)
        cutoff = stamp - self.retention
        while self._sample_stamps and self._sample_stamps[0] < cutoff:
            self._sample_stamps.pop(0)

    def nearest_sample_deviation(self, stamp: RosTime) -> Optional[Duration]:
        """Return the time gap to the closest recorded sample, if any is known yet."""
        if not self._sample_stamps:
            return None
        index = bisect.bisect_left(self._sample_stamps, stamp)
        neighbors = self._sample_stamps[max(index - 1, 0) : index + 1]
        closest_gap_ns = min(abs((stamp - neighbor).nanoseconds) for neighbor in neighbors)
        return Duration(nanoseconds=closest_gap_ns)


class TfMessageFilter(SimpleFilter):
    """tf2-aware stand-in for C++'s ``tf2_ros::MessageFilter``, which has no rclpy binding.

    Wraps an upstream :class:`message_filters.SimpleFilter`. Each incoming message group is
    queued until ``tf_buffer`` can resolve a transform from its frame to ``target_frame`` at its
    timestamp, then forwarded as ``(transform, *messages)``. Groups that wait longer than
    ``max_wait`` are dropped and reported via ``on_drop``.

    If ``max_deviation`` is set, ``tf_buffer`` must be a :class:`TfSampleTrackingBuffer`: a group
    is also dropped once resolvable if the nearest real tf sample is farther than that from its
    timestamp, since interpolating across a wider gap means no genuine pose was ever produced for
    it (e.g. an upstream SLAM system that skipped the frame).

    Call :meth:`update` once per spin iteration, since tf data can arrive independently of new
    messages on the wrapped filter.
    """

    def __init__(
        self,
        node: Node,
        upstream_filter: SimpleFilter,
        tf_buffer: Buffer,
        target_frame: str,
        source_frame_from_messages: Callable[..., str],
        stamp_from_messages: Callable[..., RosTime],
        queue_size: int = 10,
        max_wait: Duration = Duration(seconds=1.0),
        max_deviation: Optional[Duration] = None,
        on_drop: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__()
        self.node = node
        self.tf_buffer = tf_buffer
        self.target_frame = target_frame
        self.source_frame_from_messages = source_frame_from_messages
        self.stamp_from_messages = stamp_from_messages
        self.queue_size = queue_size
        self.max_wait = max_wait
        self.max_deviation = max_deviation
        self.on_drop = on_drop
        self._queue: deque[_QueueEntry] = deque()
        self.connectInput(upstream_filter)

    def connectInput(self, upstream_filter: SimpleFilter) -> None:
        self.incoming_connection = upstream_filter.registerCallback(self.add)

    def add(self, *messages: object) -> None:
        """Queue an incoming message group, dropping the oldest if the queue is full."""
        stamp = self.stamp_from_messages(*messages)
        self._queue.append((stamp, self.node.get_clock().now(), messages))
        while len(self._queue) > self.queue_size:
            _, _, dropped_messages = self._queue.popleft()
            dropped_stamp = self.stamp_from_messages(*dropped_messages)
            self._report_drop(dropped_stamp, f"queue exceeded max size {self.queue_size}")

    def update(self) -> None:
        """Release queued groups whose transform is now resolvable, and drop expired ones."""
        if not self._queue:
            return
        now = self.node.get_clock().now()
        remaining: deque[_QueueEntry] = deque()
        for stamp, enqueued_at, messages in self._queue:
            source_frame = self.source_frame_from_messages(*messages)
            transform, drop_reason = self._resolve(stamp, source_frame)
            if transform is not None:
                self.signalMessage(transform, *messages)
            elif drop_reason is not None:
                self._report_drop(stamp, drop_reason)
            elif now - enqueued_at > self.max_wait:
                reason = f"no transform from '{source_frame}' to '{self.target_frame}' within {self.max_wait}"
                self._report_drop(stamp, reason)
            else:
                remaining.append((stamp, enqueued_at, messages))
        self._queue = remaining

    def _resolve(self, stamp: RosTime, source_frame: str) -> tuple[Optional[TransformStamped], Optional[str]]:
        """Try to resolve a transform.

        Returning either (transform, None), or (None, reason) if it can never resolve
        within tolerance, or (None, None) if tf data is still pending.
        """
        if not self.tf_buffer.can_transform(self.target_frame, source_frame, stamp):
            return None, None
        if self.max_deviation is not None:
            # requires tf_buffer to be a TfSampleTrackingBuffer; see class docstring.
            deviation = self.tf_buffer.nearest_sample_deviation(stamp)  # pyright: ignore[reportAttributeAccessIssue]
            if deviation is None or deviation > self.max_deviation:
                reason = f"nearest tf sample rooted at '{self.target_frame}' is farther than {self.max_deviation} away"
                print(f"################################## {reason}", flush=True)
                return None, reason
        try:
            return self.tf_buffer.lookup_transform(self.target_frame, source_frame, stamp), None
        except ExtrapolationException:
            return None, None  # lost the bracketing sample between checks; retry next update()

    def _report_drop(self, stamp: RosTime, reason: str) -> None:
        if self.on_drop is not None:
            self.on_drop(f"Dropped message group at time {stamp}: {reason}.")
