"""
Polygonal NFC Coil Generator + Numerical Inductance Solver
----------------------------------------------------------

Generates:
1. DXF antenna geometry
2. KiCad .kicad_mod footprint
3. Matplotlib preview
4. DC trace resistance
5. Inductance using segment-by-segment Neumann integration
6. Reactance, approximate Q, and resonance with ST25DV04KC

Dependencies:
    pip install numpy ezdxf matplotlib
"""

import math
import os
from pathlib import Path

import numpy as np
import ezdxf
import matplotlib.pyplot as plt


# ============================================================
# CONSTANTS
# ============================================================

MU0 = 4 * math.pi * 1e-7
COPPER_RESISTIVITY = 1.70e-8

NFC_FREQUENCY = 13.56e6

# Nominal ST25DV04KC internal tuning capacitance
ST25DV04KC_CAPACITANCE = 28.5e-12


# ============================================================
# GEOMETRY
# ============================================================

def generate_polygon_points(
    outer_diameter_mm=31.5,
    trace_width_mm=0.3,
    spacing_mm=0.3,
    num_turns=8,
    num_sides=8,
    style=1,
):
    """
    Generate one continuous polygonal spiral path.

    outer_diameter_mm is interpreted as flat-to-flat diameter.

    Returns:
        list[(x_mm, y_mm)]
    """

    if num_sides < 3:
        raise ValueError("num_sides must be >= 3")

    degree = 180 / num_sides

    # Flat-to-flat -> vertex-to-vertex diameter
    circumscribed_diameter = (
        outer_diameter_mm
        / math.cos(math.radians(degree))
    )

    if style == 0:
        circumscribed_diameter += (
            trace_width_mm + spacing_mm
        )

    delta_r = (
        trace_width_mm + spacing_mm
    ) / math.cos(math.radians(degree))

    r_outer = (
        circumscribed_diameter
        - trace_width_mm
        / math.cos(math.radians(degree))
    ) / 2

    angle_per_section = 360 / num_sides

    angles_deg = [
        angle_per_section * side
        for side in range(num_sides)
    ]

    points = []
    r_current = r_outer

    for _turn in range(num_turns):

        for angle_deg in angles_deg:

            if style == 0:
                r_current -= delta_r / num_sides

            angle_rad = math.radians(angle_deg)

            x = r_current * math.cos(angle_rad)
            y = r_current * math.sin(angle_rad)

            points.append((x, y))

        if style == 1:
            r_current -= delta_r

    if style == 0:
        r_current -= delta_r / num_sides

    # final endpoint
    points.append((r_current, 0.0))

    return points


def points_to_segments(points):
    """
    Convert:
        P0, P1, P2 ...
    into:
        (P0,P1), (P1,P2), ...
    """

    return [
        (points[i], points[i + 1])
        for i in range(len(points) - 1)
    ]


# ============================================================
# TRACE LENGTH + DC RESISTANCE
# ============================================================

def calculate_trace_length(points):
    total_length_mm = 0.0

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        total_length_mm += math.hypot(
            x2 - x1,
            y2 - y1,
        )

    return total_length_mm


def calculate_coil_resistance(
    trace_width_mm,
    trace_thickness_mm,
    points,
):
    length_mm = calculate_trace_length(points)

    length_m = length_mm * 1e-3
    width_m = trace_width_mm * 1e-3
    thickness_m = trace_thickness_mm * 1e-3

    area = width_m * thickness_m

    resistance = (
        COPPER_RESISTIVITY
        * length_m
        / area
    )

    return resistance


# ============================================================
# INDUCTANCE SOLVER
# ============================================================

def segment_self_inductance(
    A,
    B,
    trace_width_mm,
    trace_thickness_mm,
):
    """
    Approximate self-inductance of one straight PCB trace segment.

    Uses an equivalent conductor radius based on trace width/thickness.

    Coordinates A/B are in mm.
    Returns Henries.

    This is an approximation for the singular i == j term.
    """

    A = np.asarray(A, dtype=float) * 1e-3
    B = np.asarray(B, dtype=float) * 1e-3

    length = np.linalg.norm(B - A)

    if length <= 0:
        return 0.0

    width = trace_width_mm * 1e-3
    thickness = trace_thickness_mm * 1e-3

    # crude but physically meaningful equivalent radius
    # for a rectangular cross-section
    r_eq = 0.2235 * (width + thickness)

    # keep model from becoming nonsensical on extremely short segments
    r_eq = min(r_eq, 0.25 * length)

    if r_eq <= 0:
        return 0.0

    # straight-wire partial self-inductance approximation
    L = (
        MU0
        * length
        / (2 * math.pi)
        * (
            math.log(2 * length / r_eq)
            - 1.0
        )
    )

    return max(L, 0.0)


def segment_mutual_inductance(
    A,
    B,
    C,
    D,
    order=12,
):
    """
    Numerical Neumann mutual-inductance integral between
    two different straight filament segments.

    A -> B = first segment
    C -> D = second segment

    Coordinates are in mm.
    Returns Henries.
    """

    A = np.asarray(A, dtype=float) * 1e-3
    B = np.asarray(B, dtype=float) * 1e-3
    C = np.asarray(C, dtype=float) * 1e-3
    D = np.asarray(D, dtype=float) * 1e-3

    v1 = B - A
    v2 = D - C

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)

    if l1 <= 0 or l2 <= 0:
        return 0.0

    u1 = v1 / l1
    u2 = v2 / l2

    direction_dot = np.dot(u1, u2)

    # Perpendicular ideal filament segments contribute zero
    # because dl1 . dl2 = 0.
    if abs(direction_dot) < 1e-15:
        return 0.0

    nodes, weights = np.polynomial.legendre.leggauss(order)

    s_values = 0.5 * l1 * (nodes + 1.0)
    s_weights = 0.5 * l1 * weights

    t_values = 0.5 * l2 * (nodes + 1.0)
    t_weights = 0.5 * l2 * weights

    integral = 0.0

    for s, ws in zip(s_values, s_weights):
        r1 = A + s * u1

        for t, wt in zip(t_values, t_weights):
            r2 = C + t * u2

            distance = np.linalg.norm(r1 - r2)

            # Different polygon segments can share endpoints.
            # Gauss nodes do not evaluate the endpoints, so normally
            # distance stays finite, but guard anyway.
            if distance < 1e-15:
                continue

            integral += (
                ws
                * wt
                / distance
            )

    return (
        MU0
        / (4 * math.pi)
        * direction_dot
        * integral
    )


def calculate_inductance_neumann(
    points,
    trace_width_mm,
    trace_thickness_mm,
    quadrature_order=12,
    verbose=False,
):
    """
    Compute total coil inductance as:

        L = sum(L_self_i) + 2 * sum(M_ij), i<j

    Returns:
        total_L,
        total_self,
        total_mutual
    """

    segments = points_to_segments(points)

    self_total = 0.0

    for i, (A, B) in enumerate(segments):

        L_self = segment_self_inductance(
            A,
            B,
            trace_width_mm,
            trace_thickness_mm,
        )

        self_total += L_self

        if verbose:
            print(
                f"self {i:3d}: "
                f"{L_self * 1e9:10.4f} nH"
            )

    mutual_sum_unique = 0.0

    n = len(segments)

    for i in range(n):

        A, B = segments[i]

        for j in range(i + 1, n):

            C, D = segments[j]

            M = segment_mutual_inductance(
                A,
                B,
                C,
                D,
                order=quadrature_order,
            )

            mutual_sum_unique += M

            if verbose and abs(M) > 1e-12:
                print(
                    f"M({i:3d},{j:3d}) = "
                    f"{M * 1e9:10.4f} nH"
                )

    mutual_total = 2.0 * mutual_sum_unique

    total = (
        self_total
        + mutual_total
    )

    return (
        total,
        self_total,
        mutual_total,
    )


# ============================================================
# NFC ELECTRICAL CALCULATIONS
# ============================================================

def calculate_inductive_reactance(
    inductance_h,
    frequency=NFC_FREQUENCY,
):
    return (
        2
        * math.pi
        * frequency
        * inductance_h
    )


def calculate_capacitive_reactance(
    capacitance_f=ST25DV04KC_CAPACITANCE,
    frequency=NFC_FREQUENCY,
):
    return -1 / (
        2
        * math.pi
        * frequency
        * capacitance_f
    )


def calculate_resonant_frequency(
    inductance_h,
    capacitance_f=ST25DV04KC_CAPACITANCE,
):
    return 1 / (
        2
        * math.pi
        * math.sqrt(
            inductance_h
            * capacitance_f
        )
    )


def calculate_q(
    inductance_h,
    resistance,
    frequency=NFC_FREQUENCY,
):
    """
    First-order Q using DC resistance.

    Actual Q at 13.56 MHz will be lower because R_ac > R_dc.
    """

    return (
        calculate_inductive_reactance(
            inductance_h,
            frequency,
        )
        / resistance
    )


# ============================================================
# DXF EXPORT
# ============================================================

def export_dxf(
    points,
    filename,
):
    doc = ezdxf.new(
        dxfversion="AC1027"
    )

    doc.header["$INSUNITS"] = 4

    msp = doc.modelspace()

    msp.add_lwpolyline(
        points,
        close=False,
    )

    doc.saveas(filename)

    print(f"DXF saved as: {filename}")


# ============================================================
# KICAD FOOTPRINT EXPORT
# ============================================================

def export_kicad_footprint(
    points,
    trace_width_mm,
    filename,
    footprint_name,
    pad_size_mm=0.8,
):
    """
    Export as a .kicad_mod footprint.

    Pad 1 = outer terminal
    Pad 2 = inner terminal

    The spiral itself is represented as F.Cu footprint graphics.
    """

    outer_x, outer_y = points[0]
    inner_x, inner_y = points[-1]

    out = []

    out.append(
        f'(footprint "{footprint_name}"'
    )

    out.append(
        '  (version 20240108)'
    )

    out.append(
        '  (generator "python_nfc_coil_generator")'
    )

    out.append(
        '  (layer "F.Cu")'
    )

    out.append(
        '  (attr board_only exclude_from_pos_files exclude_from_bom)'
    )

    out.append(
        '''
  (fp_text reference "AE1"
    (at 0 -2)
    (layer "F.SilkS")
    (effects
      (font
        (size 1 1)
        (thickness 0.15)
      )
    )
  )'''
    )

    out.append(
        f'''
  (fp_text value "{footprint_name}"
    (at 0 2)
    (layer "F.Fab")
    (effects
      (font
        (size 1 1)
        (thickness 0.15)
      )
    )
  )'''
    )

    for i in range(len(points) - 1):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        out.append(
            f'''
  (fp_line
    (start {x1:.6f} {y1:.6f})
    (end {x2:.6f} {y2:.6f})
    (stroke
      (width {trace_width_mm:.6f})
      (type solid)
    )
    (layer "F.Cu")
  )'''
        )

    out.append(
        f'''
  (pad "1" smd circle
    (at {outer_x:.6f} {outer_y:.6f})
    (size {pad_size_mm:.6f} {pad_size_mm:.6f})
    (layers "F.Cu" "F.Mask")
  )'''
    )

    out.append(
        f'''
  (pad "2" smd circle
    (at {inner_x:.6f} {inner_y:.6f})
    (size {pad_size_mm:.6f} {pad_size_mm:.6f})
    (layers "F.Cu" "F.Mask")
  )'''
    )

    out.append(")")

    Path(filename).write_text(
        "\n".join(out)
    )

    print(
        f"KiCad footprint saved as: "
        f"{filename}"
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_coil(points):

    x, y = zip(*points)

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        x,
        y,
        marker="o",
        markersize=2,
    )

    plt.scatter(
        [points[0][0]],
        [points[0][1]],
        s=80,
        label="Pad 1 / outer",
    )

    plt.scatter(
        [points[-1][0]],
        [points[-1][1]],
        s=80,
        label="Pad 2 / inner",
    )

    plt.axis("equal")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("NFC Antenna")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# REPORT
# ============================================================

def print_report(
    points,
    trace_width_mm,
    trace_thickness_mm,
    quadrature_order,
):

    length_mm = calculate_trace_length(
        points
    )

    resistance = calculate_coil_resistance(
        trace_width_mm,
        trace_thickness_mm,
        points,
    )

    (
        inductance,
        self_L,
        mutual_L,
    ) = calculate_inductance_neumann(
        points,
        trace_width_mm,
        trace_thickness_mm,
        quadrature_order=quadrature_order,
    )

    xl = calculate_inductive_reactance(
        inductance
    )

    xc = calculate_capacitive_reactance()

    fres = calculate_resonant_frequency(
        inductance
    )

    q = calculate_q(
        inductance,
        resistance,
    )

    residual_reactance = (
        xl + xc
    )

    print()
    print("=" * 60)
    print("ST25DV04KC NFC ANTENNA REPORT")
    print("=" * 60)

    print()
    print("GEOMETRY")
    print("-" * 60)

    print(
        f"Trace length:              "
        f"{length_mm:.3f} mm"
    )

    print(
        f"Trace width:               "
        f"{trace_width_mm:.3f} mm"
    )

    print(
        f"Copper thickness:          "
        f"{trace_thickness_mm:.3f} mm"
    )

    print(
        f"Number of line segments:   "
        f"{len(points) - 1}"
    )

    print()
    print("RESISTANCE")
    print("-" * 60)

    print(
        f"DC resistance:             "
        f"{resistance:.6f} ohm"
    )

    print()
    print("NEUMANN INDUCTANCE")
    print("-" * 60)

    print(
        f"Self contribution:         "
        f"{self_L * 1e6:.6f} uH"
    )

    print(
        f"Mutual contribution:       "
        f"{mutual_L * 1e6:.6f} uH"
    )

    print(
        f"Total estimated L:         "
        f"{inductance * 1e6:.6f} uH"
    )

    print()
    print("AT 13.56 MHz")
    print("-" * 60)

    print(
        f"XL:                        "
        f"{xl:.3f} ohm"
    )

    print(
        f"ST25DV04KC XC:             "
        f"{xc:.3f} ohm"
    )

    print(
        f"Residual reactance:        "
        f"{residual_reactance:.3f} ohm"
    )

    print(
        f"Approx coil Q (Rdc):       "
        f"{q:.2f}"
    )

    print()
    print("RESONANCE")
    print("-" * 60)

    print(
        f"Nominal chip capacitance:  "
        f"{ST25DV04KC_CAPACITANCE * 1e12:.2f} pF"
    )

    print(
        f"Ideal LC resonance:        "
        f"{fres / 1e6:.4f} MHz"
    )

    print()
    print("TARGET")
    print("-" * 60)

    print(
        "Desired region:            "
        "~4.7-4.8 uH"
    )

    print(
        "NFC carrier:               "
        "13.56 MHz"
    )

    print("=" * 60)




def generate_regular_polygon_spiral(
    outer_diameter_mm=31.5,
    trace_width_mm=0.3,
    spacing_mm=0.3,
    num_turns=8,
    num_sides=8,
):
    """
    Generate a continuous spiral based on nested regular polygons.

    outer_diameter_mm:
        Flat-to-flat OUTER dimension of the finished copper coil.

    trace_width_mm:
        Copper trace width.

    spacing_mm:
        Edge-to-edge spacing between neighboring turns.

    num_turns:
        Number of turns.

    num_sides:
        Any regular polygon with num_sides >= 3.

    Returns:
        list of (x_mm, y_mm) centerline coordinates.
    """

    if not isinstance(num_sides, int):
        raise TypeError("num_sides must be an integer")

    if num_sides < 3:
        raise ValueError("num_sides must be >= 3")

    if num_turns < 1:
        raise ValueError("num_turns must be >= 1")

    if trace_width_mm <= 0:
        raise ValueError("trace_width_mm must be > 0")

    if spacing_mm < 0:
        raise ValueError("spacing_mm must be >= 0")

    pitch = trace_width_mm + spacing_mm

    #
    # outer_diameter_mm describes the copper's outer edge.
    #
    # Therefore the centerline of the first turn is half a
    # trace width inward.
    #
    outer_apothem = outer_diameter_mm / 2.0

    first_centerline_apothem = (
        outer_apothem
        - trace_width_mm / 2.0
    )

    #
    # Check that the requested number of turns physically fits.
    #
    innermost_apothem = (
        first_centerline_apothem
        - (num_turns - 1) * pitch
    )

    if innermost_apothem <= trace_width_mm / 2:
        raise ValueError(
            "Too many turns for this outer diameter / "
            "trace width / spacing."
        )

    points = []

    angle_step = 2.0 * math.pi / num_sides

    #
    # Starting orientation:
    #
    # angle_offset = 0 means one vertex points toward +X.
    #
    angle_offset = 0.0

    for turn in range(num_turns):

        apothem = (
            first_centerline_apothem
            - turn * pitch
        )

        #
        # Regular polygon:
        #
        # apothem = R*cos(pi/n)
        #
        radius = (
            apothem
            / math.cos(math.pi / num_sides)
        )

        for vertex in range(num_sides):

            angle = (
                angle_offset
                + vertex * angle_step
            )

            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            points.append((x, y))

    #
    # Add final endpoint on innermost polygon.
    #
    final_apothem = (
        first_centerline_apothem
        - num_turns * pitch
    )

    if final_apothem > 0:

        final_radius = (
            final_apothem
            / math.cos(math.pi / num_sides)
        )

        points.append(
            (
                final_radius * math.cos(angle_offset),
                final_radius * math.sin(angle_offset),
            )
        )

    return points


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # USER PARAMETERS
    # --------------------------------------------------------

    OUTER_DIAMETER_MM = 31.5

    TRACE_WIDTH_MM = 0.30

    TRACE_THICKNESS_MM = 0.035

    SPACING_MM = 0.325

    NUM_TURNS = 12

    # Any >= 3 works geometrically.
    # 8 = octagon.
    NUM_SIDES = 12

    STYLE = 1

    PAD_SIZE_MM = 0.8

    # 8-16 is usually enough for early experiments.
    # Higher = slower but more accurate pair integration.
    QUADRATURE_ORDER = 12

    # --------------------------------------------------------
    # Generate antenna
    # --------------------------------------------------------

    points = generate_regular_polygon_spiral(
        outer_diameter_mm=
            OUTER_DIAMETER_MM,

        trace_width_mm=
            TRACE_WIDTH_MM,

        spacing_mm=
            SPACING_MM,

        num_turns=
            NUM_TURNS,

        num_sides=
            NUM_SIDES,

   )

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    base_name = (
        f"NFC_"
        f"NS{NUM_SIDES}_"
        f"OD{OUTER_DIAMETER_MM}_"
        f"NT{NUM_TURNS}_"
        f"TW{TRACE_WIDTH_MM}_"
        f"SP{SPACING_MM}"
    )

    dxf_filename = os.path.join(
        script_dir,
        base_name + ".dxf",
    )

    pretty_dir = os.path.join(
        script_dir,
        "NFC_Antennas.pretty",
    )

    os.makedirs(
        pretty_dir,
        exist_ok=True,
    )

    footprint_filename = os.path.join(
        pretty_dir,
        base_name + ".kicad_mod",
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    export_dxf(
        points,
        dxf_filename,
    )

    export_kicad_footprint(
        points,
        trace_width_mm=
            TRACE_WIDTH_MM,

        filename=
            footprint_filename,

        footprint_name=
            base_name,

        pad_size_mm=
            PAD_SIZE_MM,
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    print_report(
        points=
            points,

        trace_width_mm=
            TRACE_WIDTH_MM,

        trace_thickness_mm=
            TRACE_THICKNESS_MM,

        quadrature_order=
            QUADRATURE_ORDER,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plot_coil(points)
