#!/usr/bin/env python3
"""
Memory access pattern animations for matrix B.

Usage:
    manim -pql memory_access_anim.py RowMajorAccess
    manim -pql memory_access_anim.py ColumnMajorAccess
    manim -pql memory_access_anim.py CompareAccessPatterns
"""

from manim import *

class RowMajorAccess(Scene):
    """Show row-major access pattern - good cache performance"""
    def construct(self):
        # Title
        title = Text("Row-Major Access Pattern", font_size=40, color=GREEN)
        subtitle = Text("Accessing matrix B row-by-row (cache-friendly)", font_size=24, color=GRAY)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.wait(0.5)

        # Matrix B
        n = 4
        cell_size = 0.7
        matrix = self.create_matrix_grid(n, n, cell_size, GREEN)
        matrix_label = Text("Matrix B", font_size=28).next_to(matrix, UP, buff=0.5)
        matrix.shift(UP * 0.5 + LEFT * 3)
        matrix_label.next_to(matrix, UP, buff=0.3)

        self.play(Create(matrix), Write(matrix_label))
        self.wait(0.5)

        # Memory layout visualization
        mem_title = Text("Physical Memory Layout", font_size=24, color=BLUE)
        mem_title.shift(DOWN * 0.8)

        # Create memory cells (linear horizontal layout)
        mem_cells = VGroup()
        mem_cell_width = 0.35
        mem_cell_height = 0.5

        for i in range(n * n):
            cell = Rectangle(width=mem_cell_width, height=mem_cell_height)
            cell.set_stroke(BLUE, width=1)
            cell.set_fill(BLUE, opacity=0.1)

            # Horizontal linear layout
            x_pos = (i - (n * n - 1) / 2) * mem_cell_width
            cell.move_to([x_pos, -1.5, 0])

            # Add index label
            label = Text(str(i), font_size=10, color=GRAY)
            label.move_to(cell.get_center())

            mem_cells.add(VGroup(cell, label))

        self.play(Create(mem_cells), Write(mem_title))
        self.wait(0.5)

        # Cache line indicator
        cache_line_size = 2  # 2 elements per cache line
        cache_text = Text(f"Cache line = {cache_line_size} elements", font_size=20, color=YELLOW)
        cache_text.shift(DOWN * 2.3)
        self.play(Write(cache_text))

        # Stats
        hits_text = Text("Cache Hits: 0", font_size=24, color=GREEN).to_edge(DOWN, buff=1.5)
        misses_text = Text("Cache Misses: 0", font_size=24, color=RED).next_to(hits_text, RIGHT, buff=1)
        self.play(Write(hits_text), Write(misses_text))

        hits = 0
        misses = 0
        current_cache_line = -1

        # Animate row-major access
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                cache_line = idx // cache_line_size

                # Highlight in matrix
                matrix_cell = matrix[i * n + j]

                # Highlight in memory
                mem_cell_group = mem_cells[idx]
                mem_cell = mem_cell_group[0]

                # Check cache hit/miss
                is_hit = (cache_line == current_cache_line)

                if is_hit:
                    hits += 1
                    color = GREEN
                else:
                    misses += 1
                    color = RED
                    current_cache_line = cache_line

                # Animate
                self.play(
                    matrix_cell.animate.set_fill(color, opacity=0.7),
                    mem_cell.animate.set_fill(color, opacity=0.7),
                    run_time=0.15
                )

                # Update stats
                new_hits = Text(f"Cache Hits: {hits}", font_size=24, color=GREEN).move_to(hits_text)
                new_misses = Text(f"Cache Misses: {misses}", font_size=24, color=RED).move_to(misses_text)
                self.remove(hits_text, misses_text)
                self.add(new_hits, new_misses)
                hits_text, misses_text = new_hits, new_misses

                # Reset
                self.play(
                    matrix_cell.animate.set_fill(GREEN, opacity=0.2),
                    mem_cell.animate.set_fill(BLUE, opacity=0.1),
                    run_time=0.05
                )

        # Show final hit rate
        hit_rate = (hits / (hits + misses)) * 100
        final_text = Text(f"Hit Rate: {hit_rate:.1f}%", font_size=36, color=GREEN)
        final_text.next_to(cache_text, DOWN, buff=0.5)
        self.play(Write(final_text))
        self.wait(2)

    def create_matrix_grid(self, rows, cols, cell_size, color):
        cells = VGroup()
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(color, width=2)
                cell.set_fill(color, opacity=0.2)
                cell.move_to([j * cell_size, -i * cell_size, 0])
                cells.add(cell)
        cells.move_to(ORIGIN)
        return cells


class ColumnMajorAccess(Scene):
    """Show column-major access pattern - poor cache performance"""
    def construct(self):
        # Title
        title = Text("Column-Major Access Pattern", font_size=40, color=RED)
        subtitle = Text("Accessing matrix B column-by-column (cache-unfriendly!)", font_size=24, color=GRAY)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.wait(0.5)

        # Matrix B
        n = 4
        cell_size = 0.7
        matrix = self.create_matrix_grid(n, n, cell_size, RED)
        matrix_label = Text("Matrix B", font_size=28).next_to(matrix, UP, buff=0.5)
        matrix.shift(UP * 0.5 + LEFT * 3)
        matrix_label.next_to(matrix, UP, buff=0.3)

        self.play(Create(matrix), Write(matrix_label))
        self.wait(0.5)

        # Memory layout visualization
        mem_title = Text("Physical Memory Layout (still row-major!)", font_size=24, color=BLUE)
        mem_title.shift(DOWN * 0.8)

        # Create memory cells (linear horizontal layout)
        mem_cells = VGroup()
        mem_cell_width = 0.35
        mem_cell_height = 0.5

        for i in range(n * n):
            cell = Rectangle(width=mem_cell_width, height=mem_cell_height)
            cell.set_stroke(BLUE, width=1)
            cell.set_fill(BLUE, opacity=0.1)

            # Horizontal linear layout
            x_pos = (i - (n * n - 1) / 2) * mem_cell_width
            cell.move_to([x_pos, -1.5, 0])

            # Add index label
            label = Text(str(i), font_size=10, color=GRAY)
            label.move_to(cell.get_center())

            mem_cells.add(VGroup(cell, label))

        self.play(Create(mem_cells), Write(mem_title))
        self.wait(0.5)

        # Cache line indicator
        cache_line_size = 2
        cache_text = Text(f"Cache line = {cache_line_size} elements", font_size=20, color=YELLOW)
        cache_text.shift(DOWN * 2.3)
        self.play(Write(cache_text))

        # Stats
        hits_text = Text("Cache Hits: 0", font_size=24, color=GREEN).to_edge(DOWN, buff=1.5)
        misses_text = Text("Cache Misses: 0", font_size=24, color=RED).next_to(hits_text, RIGHT, buff=1)
        self.play(Write(hits_text), Write(misses_text))

        hits = 0
        misses = 0
        cache_lines = set()

        # Animate column-major access (this is what naive matmul does!)
        for j in range(n):
            for i in range(n):
                idx = i * n + j  # Memory index (still row-major!)
                cache_line = idx // cache_line_size

                # Highlight in matrix (column by column)
                matrix_cell = matrix[i * n + j]

                # Highlight in memory (jumping around!)
                mem_cell_group = mem_cells[idx]
                mem_cell = mem_cell_group[0]

                # Check cache hit/miss
                is_hit = (cache_line in cache_lines)

                if is_hit:
                    hits += 1
                    color = GREEN
                else:
                    misses += 1
                    color = RED
                    cache_lines.add(cache_line)

                    # Evict old cache lines (simple LRU - keep last 3)
                    if len(cache_lines) > 3:
                        cache_lines.pop()

                # Animate with emphasis on jumping memory access
                self.play(
                    matrix_cell.animate.set_fill(color, opacity=0.7),
                    mem_cell.animate.set_fill(color, opacity=0.7),
                    run_time=0.15
                )

                # Update stats
                new_hits = Text(f"Cache Hits: {hits}", font_size=24, color=GREEN).move_to(hits_text)
                new_misses = Text(f"Cache Misses: {misses}", font_size=24, color=RED).move_to(misses_text)
                self.remove(hits_text, misses_text)
                self.add(new_hits, new_misses)
                hits_text, misses_text = new_hits, new_misses

                # Reset
                self.play(
                    matrix_cell.animate.set_fill(RED, opacity=0.2),
                    mem_cell.animate.set_fill(BLUE, opacity=0.1),
                    run_time=0.05
                )

        # Show final hit rate
        hit_rate = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
        final_text = Text(f"Hit Rate: {hit_rate:.1f}%", font_size=36, color=RED)
        final_text.next_to(cache_text, DOWN, buff=0.5)
        self.play(Write(final_text))

        # Warning
        warning = Text("Each column access = new cache line!", font_size=28, color=ORANGE)
        warning.next_to(final_text, DOWN, buff=0.3)
        self.play(Write(warning))

        self.wait(2)

    def create_matrix_grid(self, rows, cols, cell_size, color):
        cells = VGroup()
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(color, width=2)
                cell.set_fill(color, opacity=0.2)
                cell.move_to([j * cell_size, -i * cell_size, 0])
                cells.add(cell)
        cells.move_to(ORIGIN)
        return cells


class CompareAccessPatterns(Scene):
    """Side-by-side comparison of row-major vs column-major access"""
    def construct(self):
        # Title
        title = Text("Memory Access Pattern Comparison", font_size=42)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        n = 4
        cell_size = 0.6

        # Left side: Row-major
        row_title = Text("Row-Major ✓", font_size=30, color=GREEN)
        row_title.shift(LEFT * 3.5 + UP * 2.5)
        row_matrix = self.create_matrix_grid(n, n, cell_size, GREEN)
        row_matrix.shift(LEFT * 3.5 + UP * 0.3)

        # Right side: Column-major
        col_title = Text("Column-Major ✗", font_size=30, color=RED)
        col_title.shift(RIGHT * 3.5 + UP * 2.5)
        col_matrix = self.create_matrix_grid(n, n, cell_size, RED)
        col_matrix.shift(RIGHT * 3.5 + UP * 0.3)

        self.play(
            Write(row_title), Write(col_title),
            Create(row_matrix), Create(col_matrix)
        )
        self.wait(0.5)

        # Stats
        row_hits_text = Text("Hits: 0", font_size=20, color=GREEN).next_to(row_matrix, DOWN, buff=0.5)
        row_misses_text = Text("Misses: 0", font_size=20, color=RED).next_to(row_hits_text, DOWN, buff=0.2)

        col_hits_text = Text("Hits: 0", font_size=20, color=GREEN).next_to(col_matrix, DOWN, buff=0.5)
        col_misses_text = Text("Misses: 0", font_size=20, color=RED).next_to(col_hits_text, DOWN, buff=0.2)

        self.play(
            Write(row_hits_text), Write(row_misses_text),
            Write(col_hits_text), Write(col_misses_text)
        )

        row_hits = 0
        row_misses = 0
        col_hits = 0
        col_misses = 0

        cache_line_size = 2
        row_current_line = -1
        col_cache_lines = set()

        # Animate both simultaneously
        for step in range(n * n):
            # Row-major coordinates
            row_i = step // n
            row_j = step % n
            row_idx = row_i * n + row_j

            # Column-major coordinates
            col_j = step // n
            col_i = step % n
            col_idx = col_i * n + col_j

            # Check row-major cache
            row_cache_line = row_idx // cache_line_size
            if row_cache_line == row_current_line:
                row_hits += 1
                row_color = GREEN
            else:
                row_misses += 1
                row_color = RED
                row_current_line = row_cache_line

            # Check column-major cache
            col_cache_line = col_idx // cache_line_size
            if col_cache_line in col_cache_lines:
                col_hits += 1
                col_color = GREEN
            else:
                col_misses += 1
                col_color = RED
                col_cache_lines.add(col_cache_line)
                if len(col_cache_lines) > 3:
                    col_cache_lines.pop()

            # Animate
            self.play(
                row_matrix[row_idx].animate.set_fill(row_color, opacity=0.7),
                col_matrix[col_idx].animate.set_fill(col_color, opacity=0.7),
                run_time=0.12
            )

            # Update stats
            new_row_hits = Text(f"Hits: {row_hits}", font_size=20, color=GREEN).move_to(row_hits_text)
            new_row_misses = Text(f"Misses: {row_misses}", font_size=20, color=RED).move_to(row_misses_text)
            new_col_hits = Text(f"Hits: {col_hits}", font_size=20, color=GREEN).move_to(col_hits_text)
            new_col_misses = Text(f"Misses: {col_misses}", font_size=20, color=RED).move_to(col_misses_text)

            self.remove(row_hits_text, row_misses_text, col_hits_text, col_misses_text)
            self.add(new_row_hits, new_row_misses, new_col_hits, new_col_misses)
            row_hits_text = new_row_hits
            row_misses_text = new_row_misses
            col_hits_text = new_col_hits
            col_misses_text = new_col_misses

            # Reset
            self.play(
                row_matrix[row_idx].animate.set_fill(GREEN, opacity=0.2),
                col_matrix[col_idx].animate.set_fill(RED, opacity=0.2),
                run_time=0.05
            )

        # Show comparison
        row_hit_rate = (row_hits / (row_hits + row_misses)) * 100
        col_hit_rate = (col_hits / (col_hits + col_misses)) * 100

        comparison = Text(
            f"Row-major: {row_hit_rate:.0f}% hit rate    Column-major: {col_hit_rate:.0f}% hit rate",
            font_size=28,
            t2c={"Row-major": GREEN, "Column-major": RED}
        )
        comparison.to_edge(DOWN, buff=0.5)
        self.play(Write(comparison))

        # Conclusion
        conclusion = Text("This is why naive matmul is slow!", font_size=32, color=YELLOW)
        conclusion.next_to(comparison, UP, buff=0.3)
        self.play(Write(conclusion))

        self.wait(3)

    def create_matrix_grid(self, rows, cols, cell_size, color):
        cells = VGroup()
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(color, width=2)
                cell.set_fill(color, opacity=0.2)
                cell.move_to([j * cell_size, -i * cell_size, 0])
                cells.add(cell)
        cells.move_to(ORIGIN)
        return cells
