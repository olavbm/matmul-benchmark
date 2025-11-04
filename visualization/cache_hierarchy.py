#!/usr/bin/env python3
"""
Cache hierarchy animation showing how a single f64 request loads an entire cache line.

Usage:
    manim -pql cache_hierarchy.py CacheLineLoad
    manim -pqh cache_hierarchy.py CacheLineLoad  # High quality
"""

from manim import *

class CacheLineLoad(Scene):
    def construct(self):
        # Title
        title = Text("Cache Hierarchy: Loading a Single f64", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create memory hierarchy layers - show ellipsis to indicate many more cache lines
        ram, ram_blocks = self.create_memory_layer("RAM", 2, 0.4, GRAY, -3, show_ellipsis=True)
        l3, l3_blocks = self.create_memory_layer("L3 Cache", 2, 0.4, RED_D, -1.8, show_ellipsis=True)
        l2, l2_blocks = self.create_memory_layer("L2 Cache", 2, 0.4, ORANGE, -0.6, show_ellipsis=True)
        l1, l1_blocks = self.create_memory_layer("L1 Cache", 2, 0.4, YELLOW, 0.6, show_ellipsis=True)
        registers, reg_blocks = self.create_memory_layer("Registers", 1, 0.4, GREEN, 1.8, show_ellipsis=False)

        # Add labels for latency
        ram_label = Text("~200 cycles", font_size=18, color=GRAY).next_to(ram, RIGHT, buff=0.3)
        l3_label = Text("~40 cycles", font_size=18, color=RED_D).next_to(l3, RIGHT, buff=0.3)
        l2_label = Text("~12 cycles", font_size=18, color=ORANGE).next_to(l2, RIGHT, buff=0.3)
        l1_label = Text("~4 cycles", font_size=18, color=YELLOW).next_to(l1, RIGHT, buff=0.3)
        reg_label = Text("~1 cycle", font_size=18, color=GREEN).next_to(registers, RIGHT, buff=0.3)

        layers = VGroup(ram, l3, l2, l1, registers)
        labels = VGroup(ram_label, l3_label, l2_label, l1_label, reg_label)

        self.play(
            *[Create(layer) for layer in layers],
            *[Write(label) for label in labels]
        )
        self.wait(1)

        # Add code request at bottom
        code_text = Text("Request: matrix[5][3]", font_size=24, color=BLUE)
        code_text.to_edge(DOWN, buff=1.5)

        self.play(Write(code_text))
        self.wait(0.5)

        # Show request propagating down to RAM (cache misses)
        # Check L1 - bright red glow to indicate miss
        self.play(l1.animate.set_color("#FF0000").set_fill(opacity=0.8), run_time=0.4)
        self.wait(0.2)

        # Check L2 - bright red glow to indicate miss
        self.play(l2.animate.set_color("#FF0000").set_fill(opacity=0.8), run_time=0.4)
        self.wait(0.2)

        # Check L3 - bright red glow to indicate miss
        self.play(l3.animate.set_color("#FF0000").set_fill(opacity=0.8), run_time=0.4)
        self.wait(0.2)

        # Finally hit RAM - bright green glow to indicate found
        self.play(ram.animate.set_color("#00FF00").set_fill(opacity=0.8), run_time=0.4)
        self.wait(1)

        # Show cache line (8 f64 values = 64 bytes) - align with RAM block
        # Create visual representation of cache line - match the width of memory blocks
        # Use the same size as the blocks in the memory layers
        cache_line = VGroup(*[
            Rectangle(width=0.4, height=0.4, color=BLUE if i == 3 else WHITE, fill_opacity=0.3 if i == 3 else 0.1)
            for i in range(8)
        ]).arrange(RIGHT, buff=0.05)
        # Position inside the first RAM block
        cache_line.move_to(ram_blocks[0].get_center())

        # Label the requested element
        requested_label = Text("Requested", font_size=14, color=BLUE)
        requested_label.next_to(cache_line[3], DOWN, buff=0.1)

        self.play(
            Create(cache_line),
            Write(requested_label)
        )
        self.wait(1)

        # Now show data propagating UP through cache hierarchy
        # Reset colors
        self.play(
            l1.animate.set_color(YELLOW),
            l2.animate.set_color(ORANGE),
            l3.animate.set_color(RED_D),
        )

        # Create moving cache line visual
        moving_line = cache_line.copy()

        # RAM -> L3 (cache line is copied to L3)
        l3_cache_line = moving_line.copy()
        self.play(
            moving_line.animate.move_to(l3_blocks[0].get_center()),
            ram.animate.set_color(GRAY),
            l3.animate.set_color(GREEN),
            run_time=1.5
        )
        self.wait(0.3)

        # L3 -> L2 (cache line is also copied to L2, stays in L3)
        l2_cache_line = moving_line.copy()
        # Keep the cache line visible in L3 immediately
        l3_cache_line.move_to(l3_blocks[0].get_center())
        self.add(l3_cache_line)
        self.play(
            moving_line.animate.move_to(l2_blocks[0].get_center()),
            l3.animate.set_color(RED_D),
            l2.animate.set_color(GREEN),
            run_time=1
        )
        self.wait(0.3)

        # L2 -> L1 (cache line is also copied to L1, stays in L2 and L3)
        # Keep the cache line visible in L2 immediately
        l2_cache_line.move_to(l2_blocks[0].get_center())
        self.add(l2_cache_line)
        self.play(
            moving_line.animate.move_to(l1_blocks[0].get_center()),
            l2.animate.set_color(ORANGE),
            l1.animate.set_color(GREEN),
            run_time=0.8
        )
        self.wait(0.5)

        # Now data is in L1! The moving_line stays there as the stored cache line
        # Now extract just the requested value from L1 -> Register
        # Only move the requested element (highlighted square) from L1 to register
        requested_elem = moving_line[3].copy()
        self.play(
            requested_elem.animate.move_to(reg_blocks[0].get_center()),
            l1.animate.set_color(GREEN),
            registers.animate.set_color(GREEN),
            run_time=0.5
        )
        self.wait(0.5)

        self.play(l1.animate.set_color(YELLOW))

        # Keep the requested element visible in register
        self.wait(1.5)

    def create_memory_layer(self, name, num_blocks, block_size, color, y_pos, show_ellipsis=False):
        """Create a visual representation of a memory layer.

        Each block represents a cache line slot.
        For visual consistency, we make blocks that can contain our cache line (8 cells).

        Args:
            show_ellipsis: If True, show [...] between blocks to indicate many more cache lines

        Returns: (layer_group, blocks_group) tuple for positioning cache lines
        """
        # Create blocks - each block will hold a cache line
        # Make blocks wide enough to visually contain 8 small cells (cache line)
        if show_ellipsis:
            # Create first and last block with ellipsis in between
            first_block = Rectangle(width=block_size * 8 + 0.35, height=0.4, color=color, fill_opacity=0.3)
            ellipsis = Text("...", font_size=24, color=color)
            last_block = Rectangle(width=block_size * 8 + 0.35, height=0.4, color=color, fill_opacity=0.3)

            blocks_group = VGroup(first_block, ellipsis, last_block).arrange(RIGHT, buff=0.3)
            # For animation purposes, we return just the actual blocks (not ellipsis)
            blocks = VGroup(first_block, last_block)
        else:
            blocks = VGroup(*[
                Rectangle(width=block_size * 8 + 0.35, height=0.4, color=color, fill_opacity=0.3)
                for _ in range(num_blocks)
            ]).arrange(RIGHT, buff=0.05)
            blocks_group = blocks

        # Add label
        label = Text(name, font_size=20, color=color)
        label.next_to(blocks_group, LEFT, buff=0.5)

        layer = VGroup(label, blocks_group)
        layer.shift(UP * y_pos)

        return layer, blocks


class SequentialVsRandomAccess(Scene):
    """Show why sequential access is better than random access."""

    def construct(self):
        title = Text("Memory Access Patterns", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create array representation
        array_size = 16
        cells = VGroup(*[
            Square(side_length=0.5, color=BLUE, fill_opacity=0.2)
            for _ in range(array_size)
        ]).arrange(RIGHT, buff=0.05)

        # Add indices
        indices = VGroup(*[
            Text(str(i), font_size=14).move_to(cell.get_center())
            for i, cell in enumerate(cells)
        ])

        array_visual = VGroup(cells, indices)
        array_visual.shift(UP * 1.5)

        array_label = Text("Array in memory", font_size=24)
        array_label.next_to(array_visual, UP, buff=0.3)

        self.play(
            Write(array_label),
            Create(array_visual)
        )
        self.wait(0.5)

        # Show cache line boundaries
        cache_lines = VGroup(*[
            SurroundingRectangle(VGroup(*cells[i:i+8]), color=YELLOW, buff=0.05)
            for i in range(0, array_size, 8)
        ])

        cache_line_label = Text("Cache lines (64 bytes = 8× f64)", font_size=20, color=YELLOW)
        cache_line_label.next_to(array_visual, DOWN, buff=0.5)

        self.play(
            Create(cache_lines),
            Write(cache_line_label)
        )
        self.wait(1)

        # Sequential access pattern
        seq_title = Text("Sequential Access: array[0], array[1], array[2]...",
                        font_size=24, color=GREEN)
        seq_title.shift(DOWN * 1)

        self.play(Write(seq_title))

        # Animate sequential access
        for i in range(8):
            self.play(
                cells[i].animate.set_fill(GREEN, opacity=0.8),
                run_time=0.2
            )

        seq_result = Text("Result: 1 cache miss, 7 cache hits!", font_size=24, color=GREEN)
        seq_result.shift(DOWN * 2)
        self.play(Write(seq_result))
        self.wait(2)

        # Clear and show random access
        self.play(
            FadeOut(seq_title),
            FadeOut(seq_result),
            *[cell.animate.set_fill(BLUE, opacity=0.2) for cell in cells]
        )

        rand_title = Text("Random Access: array[5], array[13], array[2]...",
                         font_size=24, color=RED)
        rand_title.shift(DOWN * 1)

        self.play(Write(rand_title))

        # Animate random access (jumping across cache lines)
        random_indices = [5, 13, 2, 9, 0, 15, 7]
        for idx in random_indices:
            self.play(
                cells[idx].animate.set_fill(RED, opacity=0.8),
                run_time=0.3
            )

        rand_result = Text("Result: Many cache misses! 💥", font_size=24, color=RED)
        rand_result.shift(DOWN * 2)
        self.play(Write(rand_result))
        self.wait(2)

        # Final lesson
        lesson = Text("Lesson: Access memory sequentially!", font_size=28, color=YELLOW)
        lesson.to_edge(DOWN, buff=0.3)
        self.play(Write(lesson))
        self.wait(2)
