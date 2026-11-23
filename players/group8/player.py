
import random

from core.depth import Depth
from core.player import Player
from core.action import Action
from core.seasons import Season
from math import atan2, cos, radians, sin, tau
from itertools import combinations, product
from core.snapshot import JellyfishSnapshot


class Player8(Player):
    """Group 8's jellyfish.

    Called once per living jellyfish per day. Read what the jellyfish senses
    from `snapshot`, update `snapshot.memory` in place if you want to
    remember something, and return an Action choosing today's depth.

    Do not keep per-jellyfish state on `self`: memory is the only memory.

    """

    def __init__(
        self,
        memory_size: int,
        lifetime_days: int,
        rng: random.Random
    ) -> None:
        super().__init__(memory_size, lifetime_days, rng)
        self.current_speeds = [40, 20, 10, 5]
        self.first_k_days = 4
        self.mandatory_exploration_days = 30
        self.visited_flag = 1
        self.breeding_flag = 2
        self.feeding_flag = 4
        self.predator_flag = 8

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        # Full grid needs 108 bytes; smaller budgets use a stateless fallback.
        if len(snapshot.memory) < 108:
            if snapshot.in_feeding_zone or snapshot.in_breeding_zone:
                return Action(Depth.DEEP)
            return Action(Depth.SURFACE)

        return self.act_as_per_highest_memory(snapshot)

    def act_as_per_highest_memory(
        self,
        s: JellyfishSnapshot
    ) -> Action:
        """
        Bytes 0-3: headings
        Bytes 4-7: position
        Bytes 8-107: observations for the 100 grid cells

        Each grid byte stores flags:
        1 = visited
        2 = breeding
        4 = feeding
        8 = predator
        """
        x, y = self.update_position(s)
        self.set_heading(s)
        self.update_grid_cell(s, x, y)

        # Learn all four currents before using them for navigation.
        # Offspring inherit learned headings and can skip this stage.
        heading_slots = (
            (Depth.SURFACE, 0),
            (Depth.SHALLOW, 1),
            (Depth.MEDIUM, 2),
            (Depth.DEEP, 3),
        )

        for depth, slot in heading_slots:
            if s.memory[slot] == 0:
                return Action(depth)

        # Explore only while the seasonal target is unknown.
        target_flag = (
            self.feeding_flag
            if s.season in (Season.SPRING, Season.SUMMER)
            else self.breeding_flag
        )

        inside_target = (
            s.in_feeding_zone
            if target_flag == self.feeding_flag
            else s.in_breeding_zone
        )

        if not inside_target:
            if self.estimate_zone_center(s.memory, target_flag) is None:
                depth = self.explore_grid(s, x, y)

                if depth is None:
                    depth = Depth.SURFACE

                return Action(self.avoid_predator(s, x, y, depth))

        # Alternate between feeding and breeding.
        if s.season == Season.SPRING:
            if s.in_feeding_zone:
                return self.stay_put(s, x, y, self.feeding_flag)
            else:
                return self.move_to_feeding_zone(s, x, y)

        if s.season == Season.SUMMER:
            if s.in_feeding_zone:
                return self.stay_put(s, x, y, self.feeding_flag)
            else:
                return self.move_to_feeding_zone(s, x, y)

        if s.season == Season.FALL:
            if s.in_breeding_zone:
                return self.stay_put(s, x, y, self.breeding_flag)
            else:
                return self.move_to_breeding_zone(s, x, y)

        if s.season == Season.WINTER:
            if s.in_breeding_zone:
                return self.stay_put(s, x, y, self.breeding_flag)
            else:
                return self.move_to_breeding_zone(s, x, y)

        return Action(random.choice(list(Depth)))

    @staticmethod
    def update_grid_cell(
        s: JellyfishSnapshot,
        x: float,
        y: float
    ):
        column = int((x % 1000) // 100)
        row = int((y % 1000) // 100)
        cell = row * 10 + column

        flags = 1

        if s.in_breeding_zone:
            flags |= 2

        if s.in_feeding_zone:
            flags |= 4

        if s.in_predator_zone:
            flags |= 8

        s.memory[8 + cell] |= flags

    @staticmethod
    def get_current_coordinates(m: bytearray):
        x = int.from_bytes(m[4:6], "little") / 10
        y = int.from_bytes(m[6:8], "little") / 10
        return x, y

    @staticmethod
    def do_mandatory_first_four_steps(s: JellyfishSnapshot):
        assert s.day_of_year <= 4

        if s.year == s.birth_year and s.day_of_year == 1:
            return Action(Depth.SURFACE)

        if s.year == s.birth_year and s.day_of_year == 2:
            return Action(Depth.SHALLOW)

        if s.year == s.birth_year and s.day_of_year == 3:
            return Action(Depth.MEDIUM)

        return Action(Depth.DEEP)

    @staticmethod
    def set_heading(s: JellyfishSnapshot):
        if s.depth == Depth.SURFACE:
            s.memory[0] = 1 + (round(s.heading * 255 / 360) % 255)

        if s.depth == Depth.SHALLOW:
            s.memory[1] = 1 + (round(s.heading * 255 / 360) % 255)

        if s.depth == Depth.MEDIUM:
            s.memory[2] = 1 + (round(s.heading * 255 / 360) % 255)

        if s.depth == Depth.DEEP:
            s.memory[3] = 1 + (round(s.heading * 255 / 360) % 255)

        return None

    @staticmethod
    def update_position(s: JellyfishSnapshot):
        x = int.from_bytes(s.memory[4:6], "little") / 10
        y = int.from_bytes(s.memory[6:8], "little") / 10

        is_offspring = s.birth_year > 1 or s.birth_day_of_year > 1

        if is_offspring:
            first_turn = s.age_days == 1
        else:
            first_turn = s.age_days == 0

        if not first_turn:
            dx, dy = s.velocity
            x = (x + dx) % 1000
            y = (y + dy) % 1000

        x_units = round(x * 10) % 10000
        y_units = round(y * 10) % 10000

        s.memory[4:6] = x_units.to_bytes(2, "little")
        s.memory[6:8] = y_units.to_bytes(2, "little")

        return x_units / 10, y_units / 10

    @staticmethod
    def explore_grid(
        s: JellyfishSnapshot,
        x: float,
        y: float
    ) -> Depth | None:
        heading_slots = (
            (Depth.SURFACE, 0),
            (Depth.SHALLOW, 1),
            (Depth.MEDIUM, 2),
            (Depth.DEEP, 3),
        )

        velocities = {}

        for depth, slot in heading_slots:
            angle = radians((s.memory[slot] - 1) * 360 / 255)
            velocities[depth] = (
                depth.speed * sin(angle),
                -depth.speed * cos(angle)
            )

        # Fastest pairs first.
        pairs = list(combinations(
            sorted(
                velocities,
                key=lambda current_depth: current_depth.speed,
                reverse=True
            ),
            2
        ))

        visited = 1
        targets = []

        for target_cell in range(100):
            if s.memory[8 + target_cell] & visited:
                continue

            target_x = (target_cell % 10) * 100 + 50
            target_y = (target_cell // 10) * 100 + 50

            dx = (target_x - x + 500) % 1000 - 500
            dy = (target_y - y + 500) % 1000 - 500

            targets.append((dx * dx + dy * dy, dx, dy))

        # Explore nearest unvisited cells first.
        for _, dx, dy in sorted(targets):
            for first, second in pairs:
                ax, ay = velocities[first]
                bx, by = velocities[second]

                determinant = ax * by - ay * bx

                if abs(determinant) < 1e-9:
                    continue

                best_route = None

                for wrap_x, wrap_y in product(
                    (-1000, 0, 1000),
                    repeat=2
                ):
                    tx = dx + wrap_x
                    ty = dy + wrap_y

                    days_a = (tx * by - ty * bx) / determinant
                    days_b = (ax * ty - ay * tx) / determinant

                    if days_a < -1e-9 or days_b < -1e-9:
                        continue

                    a, b = round(days_a), round(days_b)

                    if a + b == 0:
                        continue

                    error_x = a * ax + b * bx - tx
                    error_y = a * ay + b * by - ty

                    if abs(error_x) >= 50 or abs(error_y) >= 50:
                        continue

                    if best_route is None or a + b < sum(best_route):
                        best_route = (a, b)

                if best_route is not None:
                    return first if best_route[0] > 0 else second

        return None

    def stay_put(
        self,
        s: JellyfishSnapshot,
        x: float,
        y: float,
        zone_flag: int
    ) -> Action:
        target = self.estimate_zone_center(s.memory, zone_flag)
        velocities = self.current_velocities(s)

        if target is None:
            return Action(
                self.avoid_predator(s, x, y, Depth.DEEP)
            )

        # Prefer moves that stay within the estimated zone radius.
        # Among them, choose the move with the largest margin.
        options = []

        for depth, (vx, vy) in velocities.items():
            nx = (x + vx) % 1000
            ny = (y + vy) % 1000

            dist2 = (
                self.wrapped_delta(nx, target[0]) ** 2
                + self.wrapped_delta(ny, target[1]) ** 2
            )

            options.append((
                dist2 > 200 ** 2,
                dist2,
                depth
            ))

        options.sort()

        return Action(
            self.avoid_predator(
                s,
                x,
                y,
                options[0][2],
                [item[2] for item in options]
            )
        )

    def move_to_breeding_zone(
        self,
        s: JellyfishSnapshot,
        x: float,
        y: float
    ) -> Action:
        target = self.estimate_zone_center(
            s.memory,
            self.breeding_flag
        )

        if target is None:
            return Action(Depth.DEEP)

        depth = self.depth_toward(s, x, y, target)

        return Action(self.avoid_predator(s, x, y, depth))

    def move_to_feeding_zone(
        self,
        s: JellyfishSnapshot,
        x: float,
        y: float
    ) -> Action:
        target = self.estimate_zone_center(
            s.memory,
            self.feeding_flag
        )

        if target is None:
            depth = self.explore_grid(s, x, y)

            if depth is None:
                depth = Depth.SURFACE

            return Action(
                self.avoid_predator(s, x, y, depth)
            )

        depth = self.depth_toward(s, x, y, target)

        return Action(self.avoid_predator(s, x, y, depth))

    def avoid_predator(
        self,
        s: JellyfishSnapshot,
        x: float,
        y: float,
        preferred: Depth,
        alternatives=None
    ) -> Depth:
        """Apply a soft predator penalty when choosing a depth.

        Entering a known predator zone adds a 55 km penalty.
        Navigation toward the seasonal target still matters.
        """
        predator = self.estimate_zone_center(
            s.memory,
            self.predator_flag
        )

        if predator is None:
            return preferred

        velocities = self.current_velocities(s)

        target_flag = (
            self.feeding_flag
            if s.season in (Season.SPRING, Season.SUMMER)
            else self.breeding_flag
        )

        target = self.estimate_zone_center(
            s.memory,
            target_flag
        )

        inside_target = (
            s.in_feeding_zone
            if target_flag == self.feeding_flag
            else s.in_breeding_zone
        )

        preferred_vx, preferred_vy = velocities[preferred]

        def score(depth: Depth) -> float:
            vx, vy = velocities[depth]

            nx = (x + vx) % 1000
            ny = (y + vy) % 1000

            pdx = self.wrapped_delta(nx, predator[0])
            pdy = self.wrapped_delta(ny, predator[1])

            risky = pdx * pdx + pdy * pdy < 200 ** 2
            risk_cost = 55.0 if risky else 0.0

            if target is not None:
                dx = self.wrapped_delta(nx, target[0])
                dy = self.wrapped_delta(ny, target[1])

                distance = (dx * dx + dy * dy) ** 0.5

                # Avoid leaving the target zone just to avoid
                # an approximate predator estimate.
                exit_cost = (
                    250.0
                    if inside_target and distance > 200
                    else 0.0
                )

                return distance + exit_cost + risk_cost

            # While exploring, preserve the preferred move
            # unless another current offers a worthwhile detour.
            deviation = (
                (vx - preferred_vx) ** 2
                + (vy - preferred_vy) ** 2
            ) ** 0.5

            return 0.5 * deviation + risk_cost

        candidates = (
            list(alternatives)
            if alternatives is not None
            else list(Depth)
        )

        if preferred not in candidates:
            candidates.insert(0, preferred)

        return min(
            candidates,
            key=lambda depth: (
                score(depth),
                depth != preferred
            )
        )

    @staticmethod
    def wrapped_mean(values: list[float]) -> float:
        angles = [
            value * tau / 1000
            for value in values
        ]

        angle = atan2(
            sum(sin(a) for a in angles),
            sum(cos(a) for a in angles)
        )

        return (angle * 1000 / tau) % 1000

    def estimate_zone_center(
        self,
        memory: bytearray,
        zone_flag: int
    ) -> tuple[float, float] | None:
        xs, ys = [], []

        for cell, flags in enumerate(memory[8:108]):
            if flags & zone_flag:
                row, column = divmod(cell, 10)

                xs.append(column * 100 + 50)
                ys.append(row * 100 + 50)

        if not xs:
            return None

        return self.wrapped_mean(xs), self.wrapped_mean(ys)

    def best_next_depth(
        self,
        velocities: dict[Depth, tuple[float, float]],
        x: float,
        y: float,
        target: tuple[float, float]
    ) -> Depth:
        target_x, target_y = target

        def distance_after_move(depth: Depth) -> float:
            vx, vy = velocities[depth]

            dx = self.wrapped_delta(x + vx, target_x)
            dy = self.wrapped_delta(y + vy, target_y)

            return dx * dx + dy * dy

        return min(velocities, key=distance_after_move)

    @staticmethod
    def wrapped_delta(start: float, end: float) -> float:
        return (end - start + 500) % 1000 - 500

    @staticmethod
    def current_velocities(
        s: JellyfishSnapshot
    ) -> dict[Depth, tuple[float, float]]:
        slots = (
            (Depth.SURFACE, 0),
            (Depth.SHALLOW, 1),
            (Depth.MEDIUM, 2),
            (Depth.DEEP, 3),
        )

        velocities = {}

        for depth, slot in slots:
            angle = radians(
                (s.memory[slot] - 1) * 360 / 255
            )

            velocities[depth] = (
                depth.speed * sin(angle),
                -depth.speed * cos(angle)
            )

        velocities[s.depth] = s.velocity

        return velocities

    def depth_toward(
        self,
        s: JellyfishSnapshot,
        x: float,
        y: float,
        target: tuple[float, float]
    ) -> Depth:
        velocities = self.current_velocities(s)

        dx = self.wrapped_delta(x, target[0])
        dy = self.wrapped_delta(y, target[1])

        best_depth = None
        best_score = (float("inf"), float("inf"))

        for first, second in combinations(velocities, 2):
            ax, ay = velocities[first]
            bx, by = velocities[second]

            determinant = ax * by - ay * bx

            if abs(determinant) < 1e-9:
                continue

            for wrap_x, wrap_y in product(
                (-1000, 0, 1000),
                repeat=2
            ):
                tx = dx + wrap_x
                ty = dy + wrap_y

                days_a = (tx * by - ty * bx) / determinant
                days_b = (ax * ty - ay * tx) / determinant

                if days_a < -1e-9 or days_b < -1e-9:
                    continue

                a, b = round(days_a), round(days_b)

                if a + b == 0:
                    continue

                error_x = a * ax + b * bx - tx
                error_y = a * ay + b * by - ty

                error_squared = (
                    error_x * error_x
                    + error_y * error_y
                )

                score = (a + b, error_squared)

                if error_squared <= 20 ** 2 and score < best_score:
                    best_score = score
                    best_depth = first if a > 0 else second

        if best_depth is not None:
            return best_depth

        return self.best_next_depth(
            velocities,
            x,
            y,
            target
        )

    def decode_memory(self, memory: bytearray) -> str:
        x, y = self.get_current_coordinates(memory)

        lines = [
            f"Position: ({x:.1f}, {y:.1f}) km",
            f"Position bytes: x={list(memory[4:6])}, y={list(memory[6:8])}",
            "",
            "Current headings:",
        ]

        for depth, encoded in zip(Depth, memory[:4]):
            if encoded == 0:
                lines.append(f"  {depth.name}: unseen")
            else:
                heading = (encoded - 1) * 360 / 255
                lines.append(
                    f"  {depth.name}: {heading:.1f}° (byte {encoded})"
                )

        lines.extend(("", "Visited cells (row, column):"))

        zone_flags = (
            (2, "breeding"),
            (4, "feeding"),
            (8, "predator")
        )

        visited_count = 0

        for cell, flags in enumerate(memory[8:108]):
            if not flags & 1:
                continue

            visited_count += 1
            row, column = divmod(cell, 10)

            zones = [
                name
                for bit, name in zone_flags
                if flags & bit
            ]

            observation = (
                ", ".join(zones)
                if zones
                else "no zone observed"
            )

            lines.append(
                f"  ({row}, {column}): {observation}"
            )

        lines.append(f"Cells visited: {visited_count}/100")

        return "\n".join(lines)
