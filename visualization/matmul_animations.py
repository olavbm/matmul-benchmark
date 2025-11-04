#!/usr/bin/env python3
"""
Matrix multiplication animations using Manim.

Usage:
    manim -pql matmul_animations.py NaiveMatMul
    manim -pql matmul_animations.py BlockedMatMul
    manim -pql matmul_animations.py SimdMatMul
"""

from manim import *

class NaiveMatMul(Scene):
    def construct(self):
        # Title
        title = Text("Naive Matrix Multiplication", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Matrix dimensions
        n = 4  # Small for clarity
        cell_size = 0.6

        # Create matrices A, B, C
        a_matrix = self.create_matrix_grid(n, n, cell_size, BLUE)
        b_matrix = self.create_matrix_grid(n, n, cell_size, GREEN)
        c_matrix = self.create_matrix_grid(n, n, cell_size, YELLOW)

        a_label = Text("A", font_size=24).next_to(a_matrix, UP)
        b_label = Text("B", font_size=24).next_to(b_matrix, UP)
        c_label = Text("C", font_size=24).next_to(c_matrix, UP)

        # Position matrices
        a_matrix.shift(LEFT * 4)
        b_matrix.shift(LEFT * 1.5)
        c_matrix.shift(RIGHT * 1.5)

        a_label.next_to(a_matrix, UP)
        b_label.next_to(b_matrix, UP)
        c_label.next_to(c_matrix, UP)

        self.play(
            Create(a_matrix), Create(b_matrix), Create(c_matrix),
            Write(a_label), Write(b_label), Write(c_label)
        )
        self.wait(0.5)

        # Counter
        ops_text = Text("Operations: 0", font_size=24).to_edge(DOWN, buff=1)
        misses_text = Text("Cache Misses: 0", font_size=24, color=RED).next_to(ops_text, RIGHT, buff=1)
        self.play(Write(ops_text), Write(misses_text))

        ops = 0
        misses = 0

        # Animate triple loop
        for i in range(n):
            for j in range(n):
                # Highlight C[i,j] being computed
                c_cell = c_matrix[i * n + j]
                self.play(c_cell.animate.set_fill(YELLOW, opacity=0.5), run_time=0.1)

                for k in range(n):
                    ops += 1

                    # Highlight A[i,k] and B[k,j]
                    a_cell = a_matrix[i * n + k]
                    b_cell = b_matrix[k * n + j]

                    # B accessed column-wise = cache miss!
                    if k % 2 == 0:
                        misses += 1
                        self.play(
                            a_cell.animate.set_fill(BLUE, opacity=0.7),
                            b_cell.animate.set_fill(RED, opacity=0.7),
                            run_time=0.1
                        )
                    else:
                        self.play(
                            a_cell.animate.set_fill(BLUE, opacity=0.7),
                            b_cell.animate.set_fill(GREEN, opacity=0.7),
                            run_time=0.1
                        )

                    # Update counters
                    new_ops = Text(f"Operations: {ops}", font_size=24).move_to(ops_text)
                    new_misses = Text(f"Cache Misses: {misses}", font_size=24, color=RED).move_to(misses_text)
                    self.remove(ops_text, misses_text)
                    self.add(new_ops, new_misses)
                    ops_text, misses_text = new_ops, new_misses

                    # Reset highlights
                    self.play(
                        a_cell.animate.set_fill(BLUE, opacity=0.2),
                        b_cell.animate.set_fill(GREEN, opacity=0.2),
                        run_time=0.05
                    )

                # Mark C[i,j] as complete
                self.play(c_cell.animate.set_fill(YELLOW, opacity=0.3), run_time=0.1)

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


class BlockedMatMul(Scene):
    def construct(self):
        # Title
        title = Text("Cache-Blocked Matrix Multiplication", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        subtitle = Text("4×4 blocks fit in L1 cache", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        self.play(Write(subtitle))
        self.wait(0.5)

        # Matrix dimensions
        n = 8  # Larger to show blocks
        block_size = 4
        cell_size = 0.4

        # Create matrices
        a_matrix = self.create_matrix_grid_with_blocks(n, n, block_size, cell_size, BLUE)
        b_matrix = self.create_matrix_grid_with_blocks(n, n, block_size, cell_size, GREEN)
        c_matrix = self.create_matrix_grid_with_blocks(n, n, block_size, cell_size, YELLOW)

        a_label = Text("A", font_size=24).next_to(a_matrix, UP)
        b_label = Text("B", font_size=24).next_to(b_matrix, UP)
        c_label = Text("C", font_size=24).next_to(c_matrix, UP)

        # Position matrices
        a_matrix.shift(LEFT * 4.5)
        b_matrix.shift(LEFT * 0.5)
        c_matrix.shift(RIGHT * 3.5)

        a_label.next_to(a_matrix, UP)
        b_label.next_to(b_matrix, UP)
        c_label.next_to(c_matrix, UP)

        self.play(
            Create(a_matrix), Create(b_matrix), Create(c_matrix),
            Write(a_label), Write(b_label), Write(c_label)
        )
        self.wait(0.5)

        # Counter
        misses_text = Text("Cache Misses: 0", font_size=24, color=RED).to_edge(DOWN, buff=1)
        self.play(Write(misses_text))

        misses = 0

        # Animate block processing
        num_blocks = n // block_size

        for bi in range(num_blocks):
            for bj in range(num_blocks):
                for bk in range(num_blocks):
                    # Highlight blocks
                    a_block = self.get_block(a_matrix, bi, bk, block_size, n)
                    b_block = self.get_block(b_matrix, bk, bj, block_size, n)
                    c_block = self.get_block(c_matrix, bi, bj, block_size, n)

                    # Only one cache miss per block!
                    misses += 1

                    self.play(
                        a_block.animate.set_fill(BLUE, opacity=0.7),
                        b_block.animate.set_fill(GREEN, opacity=0.7),
                        c_block.animate.set_fill(YELLOW, opacity=0.7),
                        run_time=0.3
                    )

                    # Update counter
                    new_misses = Text(f"Cache Misses: {misses}", font_size=24, color=RED).move_to(misses_text)
                    self.remove(misses_text)
                    self.add(new_misses)
                    misses_text = new_misses

                    self.wait(0.2)

                    # Reset highlights
                    self.play(
                        a_block.animate.set_fill(BLUE, opacity=0.2),
                        b_block.animate.set_fill(GREEN, opacity=0.2),
                        c_block.animate.set_fill(YELLOW, opacity=0.3),
                        run_time=0.2
                    )

        self.wait(2)

    def create_matrix_grid_with_blocks(self, rows, cols, block_size, cell_size, color):
        cells = VGroup()
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(color, width=1)
                cell.set_fill(color, opacity=0.2)
                cell.move_to([j * cell_size, -i * cell_size, 0])
                cells.add(cell)

        # Add block boundaries
        for i in range(0, rows + 1, block_size):
            for j in range(0, cols + 1, block_size):
                if i < rows and j < cols:
                    # Draw thicker lines for block boundaries
                    if i % block_size == 0:
                        line = Line(
                            [j * cell_size - cell_size/2, -i * cell_size + cell_size/2, 0],
                            [(j + block_size) * cell_size - cell_size/2, -i * cell_size + cell_size/2, 0],
                            color=WHITE, stroke_width=3
                        )
                        cells.add(line)

        cells.move_to(ORIGIN)
        return cells

    def get_block(self, matrix, block_i, block_j, block_size, n):
        """Get cells in a specific block"""
        block = VGroup()
        for i in range(block_size):
            for j in range(block_size):
                idx = (block_i * block_size + i) * n + (block_j * block_size + j)
                if idx < len(matrix):
                    cell = matrix[idx]
                    if isinstance(cell, Square):  # Skip lines
                        block.add(cell)
        return block


class DotProductMatMul(Scene):
    """Shows how each row of A dots with each column of B to produce C[i,j]"""
    def construct(self):
        # Title
        title = Text("Matrix Multiplication: Row × Column", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Example 2x2 matrices with actual numbers
        a_vals = [[1, 2], [3, 4]]
        b_vals = [[5, 6], [7, 8]]

        # Create matrix B (top)
        b_matrix = Matrix(b_vals, h_buff=1.2)
        b_label = MathTex("B", "=").next_to(b_matrix, LEFT)
        b_group = VGroup(b_label, b_matrix)
        b_group.shift(RIGHT * 1 + UP * 2)

        # Create matrix A (left)
        a_matrix = Matrix(a_vals, h_buff=1.2)
        a_label = MathTex("A", "=").next_to(a_matrix, LEFT)
        a_group = VGroup(a_label, a_matrix)
        a_group.shift(LEFT * 2.5 + DOWN * 0.5)

        # Show A and B
        self.play(Write(b_group), Write(a_group))
        self.wait(1)

        # Create matrix C (initially empty with placeholders)
        # Position C to align with A vertically and B horizontally
        c_matrix = Matrix([["?", "?"], ["?", "?"]], h_buff=1.2)
        # Align C's center with A's horizontal center and B's vertical center
        c_matrix.move_to([b_matrix.get_center()[0], a_matrix.get_center()[1], 0])
        c_label = MathTex("= C").next_to(c_matrix, RIGHT)

        # Fade in C
        self.play(FadeIn(c_matrix), FadeIn(c_label))
        self.wait(0.5)

        # Store computed values
        c_vals = [["?", "?"], ["?", "?"]]

        # Compute each element of C
        for i in range(2):
            for j in range(2):
                # Highlight row i of A and column j of B
                row_rect = self.create_row_highlight_for_matrix(a_matrix, i, BLUE)
                col_rect = self.create_col_highlight_for_matrix(b_matrix, j, GREEN)

                self.play(Create(row_rect), Create(col_rect), run_time=0.4)
                self.wait(0.3)

                # Show the computation formula with actual numbers
                result = 0
                for k in range(2):
                    result += a_vals[i][k] * b_vals[k][j]

                # Build the expression: C[i,j] = 1·5 + 2·7 = 19
                expr_parts = [f"C[{i},{j}]", "="]
                for k in range(2):
                    if k > 0:
                        expr_parts.append("+")
                    expr_parts.append(f"{a_vals[i][k]}")
                    expr_parts.append("\\cdot")
                    expr_parts.append(f"{b_vals[k][j]}")
                expr_parts.extend(["=", f"{result}"])

                formula = MathTex(*expr_parts, font_size=36)
                formula.next_to(c_matrix, DOWN, buff=0.8)

                self.play(Write(formula), run_time=0.8)
                self.wait(0.5)

                # Update C matrix with the result
                c_vals[i][j] = str(result)
                new_c_matrix = Matrix(c_vals, h_buff=1.2)
                new_c_matrix.move_to(c_matrix.get_center())

                self.play(Transform(c_matrix, new_c_matrix), run_time=0.4)
                self.wait(0.3)

                # Remove highlights and formula
                self.play(
                    FadeOut(row_rect),
                    FadeOut(col_rect),
                    FadeOut(formula),
                    run_time=0.3
                )
                self.wait(0.2)

        # Add the relationship "= A × B" after the existing label
        relationship = MathTex("= A \\times B").next_to(c_label, RIGHT, buff=0.2)
        self.play(FadeIn(relationship), run_time=0.5)

        self.wait(2)

    def create_row_highlight_for_matrix(self, matrix, row, color):
        """Create a rectangle highlighting a row in a Manim Matrix"""
        # Get the matrix entries
        entries = matrix.get_entries()
        n_cols = 2

        # Get positions of row elements
        row_entries = [entries[row * n_cols + col] for col in range(n_cols)]

        # Create surrounding rectangle
        rect = SurroundingRectangle(VGroup(*row_entries), color=color, buff=0.15)
        rect.set_stroke(width=3)
        return rect

    def create_col_highlight_for_matrix(self, matrix, col, color):
        """Create a rectangle highlighting a column in a Manim Matrix"""
        # Get the matrix entries
        entries = matrix.get_entries()
        n_rows = 2
        n_cols = 2

        # Get positions of column elements
        col_entries = [entries[row * n_cols + col] for row in range(n_rows)]

        # Create surrounding rectangle
        rect = SurroundingRectangle(VGroup(*col_entries), color=color, buff=0.15)
        rect.set_stroke(width=3)
        return rect


class SimdMatMul(Scene):
    def construct(self):
        # Title
        title = Text("SIMD Matrix Multiplication", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        subtitle = Text("AVX2: 4 × f64 processed in parallel", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        self.play(Write(subtitle))
        self.wait(0.5)

        # Show 4 parallel lanes
        lanes = VGroup()
        for i in range(4):
            y_pos = 1.5 - i * 1.2

            # A element
            a_box = Square(side_length=0.5).set_fill(BLUE, opacity=0.7).set_stroke(BLUE, width=2)
            a_box.shift(LEFT * 4 + UP * y_pos)
            a_text = Text(f"a[i][k+{i}]", font_size=18).move_to(a_box)

            # × symbol
            times = Text("×", font_size=24).next_to(a_box, RIGHT, buff=0.3)

            # B element
            b_box = Square(side_length=0.5).set_fill(GREEN, opacity=0.7).set_stroke(GREEN, width=2)
            b_box.next_to(times, RIGHT, buff=0.3)
            b_text = Text(f"b[k+{i}][j]", font_size=18).move_to(b_box)

            # = symbol
            equals = Text("=", font_size=24).next_to(b_box, RIGHT, buff=0.3)

            # Result
            c_box = Square(side_length=0.5).set_fill(YELLOW, opacity=0.7).set_stroke(YELLOW, width=2)
            c_box.next_to(equals, RIGHT, buff=0.3)

            lane = VGroup(a_box, a_text, times, b_box, b_text, equals, c_box)
            lanes.add(lane)

        self.play(Create(lanes))
        self.wait(0.5)

        # Highlight: ALL HAPPEN SIMULTANEOUSLY
        simultaneous = Text("ALL 4 OPERATIONS HAPPEN SIMULTANEOUSLY!", font_size=28, color=RED)
        simultaneous.to_edge(DOWN, buff=1)

        self.play(Write(simultaneous))

        # Flash all lanes
        for _ in range(3):
            self.play(*[lane.animate.set_opacity(1) for lane in lanes], run_time=0.3)
            self.play(*[lane.animate.set_opacity(0.7) for lane in lanes], run_time=0.3)

        self.wait(1)

        # Show speedup
        speedup = Text("4x faster than scalar!", font_size=32, color=GREEN)
        speedup.next_to(simultaneous, UP, buff=0.5)
        self.play(Write(speedup))

        self.wait(2)
