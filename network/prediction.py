"""
NetworkPredictionSystem - 客户端预测 / 服务器回滚 / 输入缓冲 / 状态同步 / 延迟补偿。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math


@dataclass
class State:
    """通用状态容器"""
    values: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def has(self, key: str) -> bool:
        return key in self.values


DEFAULT_CONFIG: Dict[str, Any] = {
    # 预测
    "smoothing_factor": 0.2,        # 平滑插值系数 (0, 1]
    "max_prediction_error": 10.0,   # 允许的累计预测误差上限（超过则钳制）
    # 回滚
    "max_rollback_time": 0.5,       # 单次回滚最大时间跨度（秒）
    # 输入缓冲
    "input_buffer_size": 32,        # 每个玩家最多缓存的输入条数
    "input_timeout": 1.0,           # 输入最长保留时间（秒）
    # 状态同步
    "sync_interval_min": 0.05,      # 高速移动时的同步间隔（秒）
    "sync_interval_max": 1.0,       # 静止时的同步间隔（秒）
    "sync_speed_ref": 10.0,         # 达到该速度即按最小间隔同步
    # 延迟补偿
    "max_lag_compensation": 0.3,    # 延迟补偿上限（秒）
    "jitter_window": 10,            # 延迟抖动统计窗口
}


def _vec_add(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _vec_sub(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _vec_scale(a: Tuple[float, ...], s: float) -> Tuple[float, ...]:
    return tuple(x * s for x in a)


def _vec_len(a: Tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _vec_lerp(a: Tuple[float, ...], b: Tuple[float, ...], t: float) -> Tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class NetworkPredictionSystem:
    """网络预测系统主类。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = State()
        self.history: List[Dict[str, Any]] = []
        self._initialized = False
        self._tick_count = 0
        # 每个玩家的运行时数据，输入缓冲严格按玩家隔离
        self._players: Dict[str, Dict[str, Any]] = {}

    def _cfg(self, key: str) -> Any:
        """读取配置项，缺失时回退到默认值（不修改用户传入的 config）。"""
        return self.config.get(key, DEFAULT_CONFIG[key])

    # ------------------------------------------------------------------
    # 基础生命周期
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """初始化系统"""
        self.state = State()
        self.history = []
        self._players = {}
        self._initialized = True
        return True

    @property
    def timestamp(self) -> float:
        """当前系统时间戳"""
        return self.state.timestamp

    def tick(self, delta_time: float = 1.0) -> None:
        """推进一个时间步"""
        if not self._initialized:
            self.initialize()
        self._tick_count += 1
        self.state.timestamp += delta_time
        self._update_state(delta_time)

    def _update_state(self, delta_time: float) -> None:
        """更新内部状态 - 子类可重写"""

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        return dict(self.state.values)

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return list(self.history)

    def reset(self) -> None:
        """重置系统"""
        self.state = State()
        self.history = []
        self._players = {}
        self._initialized = False
        self._tick_count = 0

    # ------------------------------------------------------------------
    # 玩家内部数据
    # ------------------------------------------------------------------
    def _player(self, player_id: str) -> Dict[str, Any]:
        return self._players.setdefault(player_id, {
            "position": (0.0, 0.0, 0.0),
            "velocity": (0.0, 0.0, 0.0),
            "acceleration": (0.0, 0.0, 0.0),
            "direction": (1.0, 0.0, 0.0),
            "anim_state": "idle",
            "predicted_error": 0.0,     # 累计预测误差
            "input_buffer": [],         # [(input, timestamp)]
            "last_sync_time": None,
            "last_sync_position": (0.0, 0.0, 0.0),
            "latency_samples": [],      # 最近的延迟样本，用于抖动估计
        })

    # ------------------------------------------------------------------
    # 客户端预测：位置 + 速度 + 加速度，带平滑与误差限制
    # ------------------------------------------------------------------
    def client_predict(
        self,
        player_id: str,
        input_cmd: Dict[str, Any],
        delta_time: float,
    ) -> Dict[str, Any]:
        """
        根据输入预测下一帧状态。

        - 用匀加速运动学积分位置/速度/加速度，停止输入时速度归零，不会继续滑行；
        - 输出与当前状态做平滑插值，服务器校正时不会瞬移；
        - 累计预测误差超过 max_prediction_error 时钳制，避免小误差滚成大瞬移。
        """
        player = self._player(player_id)
        dt = max(delta_time, 0.0)

        accel = tuple(input_cmd.get("acceleration", (0.0, 0.0, 0.0)))
        velocity = _vec_add(player["velocity"], _vec_scale(accel, dt))
        if input_cmd.get("stop"):
            # 玩家停止移动：速度清零，位置保持不动，不再往前滑
            velocity = (0.0,) * len(velocity)
            accel = (0.0,) * len(accel)
            position = player["position"]
        else:
            position = _vec_add(
                player["position"],
                _vec_add(_vec_scale(player["velocity"], dt), _vec_scale(accel, 0.5 * dt * dt)),
            )

        # 平滑：向目标状态插值，避免校正时瞬移；显式停止时不做平滑，硬停
        smoothing = min(max(self._cfg("smoothing_factor"), 0.0), 1.0)
        if not input_cmd.get("stop") and smoothing < 1.0:
            position = _vec_lerp(player["position"], position, smoothing)
            velocity = _vec_lerp(player["velocity"], velocity, smoothing)

        # 误差累积限制
        step_error = _vec_len(_vec_sub(position, player["position"]))
        player["predicted_error"] = min(
            player["predicted_error"] + step_error,
            self._cfg("max_prediction_error"),
        )

        player["position"] = position
        player["velocity"] = velocity
        player["acceleration"] = accel
        if _vec_len(velocity) > 1e-9:
            player["direction"] = _vec_scale(velocity, 1.0 / _vec_len(velocity))

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": accel,
            "direction": player["direction"],
            "accumulated_error": player["predicted_error"],
        }

    # ------------------------------------------------------------------
    # 服务器校正回滚：回滚动画与状态，检查时间戳，限制回滚幅度
    # ------------------------------------------------------------------
    def server_reconcile(
        self,
        player_id: str,
        server_state: Dict[str, Any],
        server_timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        """
        用服务器权威状态校正客户端。

        - 位置、速度、动画等整体状态一起回滚，不会只回位置而动画还在播；
        - 丢弃过期的服务器状态（时间戳不新于本地），不用旧状态覆盖新状态；
        - 单次回滚跨度超过 max_rollback_time 时拒绝，避免被拉回出生点。
        """
        player = self._player(player_id)
        local_timestamp = player.get("last_state_timestamp", 0.0)

        if server_timestamp <= local_timestamp:
            return None  # 过期状态，忽略

        if server_timestamp - local_timestamp > self._cfg("max_rollback_time") and local_timestamp > 0.0:
            return None  # 回滚幅度过大，拒绝

        for key in ("position", "velocity", "acceleration", "direction", "anim_state"):
            if key in server_state:
                player[key] = server_state[key]
        player["last_state_timestamp"] = server_timestamp
        player["predicted_error"] = 0.0  # 权威状态到达，误差清零

        self.history.append({
            "event": "reconcile",
            "player_id": player_id,
            "timestamp": server_timestamp,
        })
        return {
            "position": player["position"],
            "velocity": player["velocity"],
            "anim_state": player["anim_state"],
        }

    # ------------------------------------------------------------------
    # 输入缓冲：按玩家隔离，有大小限制与超时
    # ------------------------------------------------------------------
    def buffer_input(
        self,
        player_id: str,
        input_cmd: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        缓存玩家输入并返回该玩家当前可执行的输入列表。

        - 缓冲按 player_id 隔离，A 玩家的输入不会用到 B 玩家身上；
        - 超过 input_buffer_size 时丢弃最旧的输入，不会攒一堆一次性执行；
        - 超过 input_timeout 的输入直接过期丢弃。
        """
        player = self._player(player_id)
        now = self.state.timestamp if timestamp is None else timestamp
        buffer: List[Tuple[Dict[str, Any], float]] = player["input_buffer"]

        # 先清理超时输入，再追加新输入
        timeout = self._cfg("input_timeout")
        buffer[:] = [(cmd, ts) for cmd, ts in buffer if now - ts <= timeout]
        buffer.append((input_cmd, now))

        max_size = self._cfg("input_buffer_size")
        if len(buffer) > max_size:
            del buffer[: len(buffer) - max_size]

        return [cmd for cmd, _ in buffer]

    # ------------------------------------------------------------------
    # 状态同步：同步速度/方向，带插值与动态频率
    # ------------------------------------------------------------------
    def sync_state(
        self,
        player_id: str,
        remote_state: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        同步其他玩家的状态。

        - 位置、速度、方向一起同步，其他玩家看到的角色不再滑着走；
        - 返回从上一次同步状态到当前状态的插值系数，渲染端可平滑过渡；
        - 同步间隔随速度动态调整：静止用长间隔省带宽，高速用短间隔。
        """
        player = self._player(player_id)
        now = self.state.timestamp if timestamp is None else timestamp

        speed = _vec_len(tuple(remote_state.get("velocity", (0.0, 0.0, 0.0))))
        interval = self._sync_interval(speed)
        last = player["last_sync_time"]
        if last is not None and now - last < interval:
            return None  # 还没到该玩家的同步时刻

        previous_position = player["position"]
        for key in ("position", "velocity", "direction"):
            if key in remote_state:
                player[key] = remote_state[key]
        player["last_sync_time"] = now
        player["last_sync_position"] = previous_position

        return {
            "position": player["position"],
            "velocity": player["velocity"],
            "direction": player["direction"],
            "interp_from": previous_position,
            "interp_duration": interval,
            "next_sync_interval": interval,
        }

    def _sync_interval(self, speed: float) -> float:
        """速度越快同步越频繁，静止时用最长的省带宽间隔。"""
        lo = self._cfg("sync_interval_min")
        hi = self._cfg("sync_interval_max")
        ref = max(self._cfg("sync_speed_ref"), 1e-9)
        ratio = min(max(speed / ref, 0.0), 1.0)
        return hi - (hi - lo) * ratio

    # ------------------------------------------------------------------
    # 延迟补偿：考虑延迟波动，有上限，全程记录可回放
    # ------------------------------------------------------------------
    def lag_compensation(
        self,
        player_id: str,
        latency: float,
    ) -> Dict[str, Any]:
        """
        根据延迟样本计算补偿时长。

        - 用滑动窗口估计延迟抖动（波动），抖动大时补偿更保守；
        - 补偿时长不超过 max_lag_compensation，高延迟玩家打不到过久远的目标；
        - 每次补偿都写入 history，可回放验证。
        """
        player = self._player(player_id)
        samples: List[float] = player["latency_samples"]
        samples.append(latency)
        window = self._cfg("jitter_window")
        if len(samples) > window:
            del samples[: len(samples) - window]

        mean = sum(samples) / len(samples)
        jitter = math.sqrt(sum((s - mean) ** 2 for s in samples) / len(samples))

        # 补偿 = 平均延迟 + 抖动余量，再钳制到上限
        compensation = min(mean + jitter, self._cfg("max_lag_compensation"))

        record = {
            "event": "lag_compensation",
            "player_id": player_id,
            "latency": latency,
            "mean_latency": mean,
            "jitter": jitter,
            "compensation": compensation,
            "timestamp": self.state.timestamp,
        }
        self.history.append(record)
        return record
