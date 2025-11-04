#!/usr/bin/env python3
"""
Manim animation showing the hierarchy: scalar → vector → matrix
The same visual elements transform smoothly with brackets appearing
"""

from manim import *

class NumberHierarchy(Scene):
    def construct(self):
        # Title
        title = Text("Number Structure Hierarchy", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # STEP 1: Start with a single scalar
        scalar_label = Text("Scalar", font_size=36, color=BLUE)
        scalar_label.to_edge(UP).shift(DOWN * 0.8)

        # Single number
        num1 = MathTex("5", font_size=60, color=BLUE)
        num1.move_to(ORIGIN)

        self.play(FadeIn(scalar_label))
        self.play(Write(num1))
        self.wait(1.5)

        # STEP 2: Add more numbers beside it, then add vector brackets
        vector_label = Text("Vector = Multiple Scalars", font_size=36, color=GREEN)
        vector_label.to_edge(UP).shift(DOWN * 0.8)

        # Create 3 more numbers
        num2 = MathTex("3", font_size=60, color=GREEN)
        num3 = MathTex("7", font_size=60, color=GREEN)
        num4 = MathTex("2", font_size=60, color=GREEN)

        # Position them in a row (tighter spacing)
        numbers = VGroup(num1, num2, num3, num4)
        numbers.arrange(RIGHT, buff=0.5)
        numbers.move_to(ORIGIN)

        # Calculate where num1 needs to move (left position in the final arrangement)
        target_pos = numbers[0].get_center()

        # Change label and color of first number, and smoothly shift it left
        self.play(
            Transform(scalar_label, vector_label),
            num1.animate.set_color(GREEN).move_to(target_pos),
        )

        # Add the other numbers beside it (they're already positioned correctly)
        self.play(
            LaggedStart(*[
                FadeIn(num, shift=LEFT*0.2) for num in [num2, num3, num4]
            ], lag_ratio=0.2),
            run_time=1.0
        )
        self.wait(1)

        # Now add the vector brackets around them
        left_bracket = MathTex(r"[", font_size=80, color=GREEN)
        right_bracket = MathTex(r"]", font_size=80, color=GREEN)

        left_bracket.next_to(numbers, LEFT, buff=0.1)
        right_bracket.next_to(numbers, RIGHT, buff=0.1)

        self.play(
            GrowFromCenter(left_bracket),
            GrowFromCenter(right_bracket),
        )
        self.wait(1.5)

        # STEP 3: Stack the vector to make more rows (matrix)
        matrix_label = Text("Matrix = Multiple Vectors", font_size=36, color=RED)
        matrix_label.to_edge(UP).shift(DOWN * 0.8)

        # Create second row (tighter horizontal spacing)
        row2 = VGroup(*[
            MathTex(str(n), font_size=60, color=RED)
            for n in [1, 4, 9, 6]
        ])
        row2.arrange(RIGHT, buff=0.5)

        # Create third row (tighter horizontal spacing)
        row3 = VGroup(*[
            MathTex(str(n), font_size=60, color=RED)
            for n in [8, 2, 0, 3]
        ])
        row3.arrange(RIGHT, buff=0.5)

        # Position rows vertically (tighter vertical spacing)
        all_rows = VGroup(numbers, row2, row3)
        all_rows.arrange(DOWN, buff=0.3)
        all_rows.move_to(ORIGIN)

        # Change label and colors
        self.play(Transform(scalar_label, matrix_label))
        self.play(
            *[num.animate.set_color(RED) for num in numbers],
            left_bracket.animate.set_color(RED),
            right_bracket.animate.set_color(RED),
        )

        # Add row 2 - smoothly shift row 1 up and fade in row 2
        self.play(
            numbers.animate.shift(UP * 0.4),
            left_bracket.animate.shift(UP * 0.4),
            right_bracket.animate.shift(UP * 0.4),
            LaggedStart(*[FadeIn(num, shift=UP*0.2) for num in row2], lag_ratio=0.1),
            run_time=1.0
        )
        self.wait(0.5)

        # Add row 3 - smoothly shift everything up and fade in row 3
        self.play(
            numbers.animate.shift(UP * 0.4),
            row2.animate.shift(UP * 0.4),
            left_bracket.animate.shift(UP * 0.4),
            right_bracket.animate.shift(UP * 0.4),
            LaggedStart(*[FadeIn(num, shift=UP*0.2) for num in row3], lag_ratio=0.1),
            run_time=1.0
        )
        self.wait(1)

        # Grow the brackets to encompass all rows
        # Use proper scaling to make them tall enough
        matrix_height = all_rows.height
        new_left = MathTex(r"\left[\begin{array}{c} \\ \\ \\ \end{array}\right.", font_size=60, color=RED)
        new_right = MathTex(r"\left.\begin{array}{c} \\ \\ \\ \end{array}\right]", font_size=60, color=RED)

        # Scale to match matrix height
        new_left.stretch_to_fit_height(matrix_height + 0.4)
        new_right.stretch_to_fit_height(matrix_height + 0.4)

        new_left.next_to(all_rows, LEFT, buff=0.15)
        new_right.next_to(all_rows, RIGHT, buff=0.15)

        self.play(
            Transform(left_bracket, new_left),
            Transform(right_bracket, new_right),
            run_time=0.8
        )
        self.wait(1.5)

        # STEP 4: Highlight each row to show "each row is a vector"
        highlight_label = Text("Each row is a vector", font_size=28, color=YELLOW)
        highlight_label.to_edge(DOWN).shift(UP * 0.5)

        self.play(Write(highlight_label))
        for row in [numbers, row2, row3]:
            self.play(
                *[num.animate.set_opacity(1.0).scale(1.2) for num in row],
                run_time=0.4
            )
            self.play(
                *[num.animate.set_opacity(1.0).scale(1/1.2) for num in row],
                run_time=0.4
            )

        self.play(FadeOut(highlight_label))
        self.wait(2)

        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
