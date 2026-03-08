#!/usr/bin/env python3
"""
generate_dataset.py
===================
Генерирует тренировочный датасет папок с симуляциями MuMax3 для
обучения E(3)-эквивариантных GNN. Каждая папка содержит:
  - sim.mx3   : валидный входной скрипт MuMax3
  - geometry_info.json : параметры + аналитическое уравнение поверхности

После создания всех папок в корневой директории записывается единый
файл ``tasks.txt`` (один путь к директории на строку) для запуска
через SLURM Job Array.

Использование
-------------
    python3 generate_dataset.py                        # 100 образцов → ./dataset
    python3 generate_dataset.py --num_samples 5 \\
                                --output_dir test_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import textwrap
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Физические константы, используемые в каждой симуляции
# ---------------------------------------------------------------------------
CELL_SIZE_M = 2e-9          # размер ячейки нм
PADDING_M   = 16e-9         # отступ нм с каждой стороны
PADDING_CELLS = int(PADDING_M / CELL_SIZE_M)  # количество ячеек с каждй стороны

AEX   = 1.3e-11   # Дж/м  – константа обменного взаимодействия
MSAT  = 8e5        # А/м   – намагниченность насыщения
ALPHA = 0.5        # –     – затухание Гильберта


# ---------------------------------------------------------------------------
# Вспомогательные функции для вычисления сетки, удобной для БПФ (FFT)
# ---------------------------------------------------------------------------
def _is_fft_friendly(n: int) -> bool:
    """Возвращает True, если *n* имеет только малые простые множители (2, 3, 5, 7)."""
    if n <= 0:
        return False
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def round_to_fft_friendly(n: int) -> int:
    """Округляет *n* **вверх** до ближайшего целого числа, удобного для БПФ (≥ n)."""
    candidate = n
    while not _is_fft_friendly(candidate):
        candidate += 1
    return candidate


def compute_grid(bbox_x: float, bbox_y: float, bbox_z: float):
    """
    Принимая размеры ограничивающего параллелепипеда объекта (в метрах),
    возвращает (Nx, Ny, Nz) после добавления отступа *PADDING_M* с каждой
    стороны и округления до значений, удобных для БПФ.
    """
    sizes = []
    for dim in (bbox_x, bbox_y, bbox_z):
        total = dim + 2 * PADDING_M
        n_cells = max(1, math.ceil(total / CELL_SIZE_M))
        sizes.append(round_to_fft_friendly(n_cells))
    return tuple(sizes)


# ===================================================================
# Генераторы геометрии
# Каждый возвращает (mx3_shape_expr, params_dict, surface_equation_str)
# ===================================================================

def generate_nanotube(rng: random.Random):
    """Полый цилиндр (нанотрубка) вдоль оси z."""
    R_outer_nm = rng.uniform(20, 60)
    wall_nm    = rng.uniform(4, min(15, R_outer_nm - 4))
    length_nm  = rng.uniform(60, 200)

    R_outer = R_outer_nm * 1e-9
    R_inner = (R_outer_nm - wall_nm) * 1e-9
    L       = length_nm * 1e-9

    # MuMax3: Cylinder(diameter, height)
    outer_diam = f"{2*R_outer:.6e}"
    inner_diam = f"{2*R_inner:.6e}"
    height     = f"{L:.6e}"

    shape_expr = (
        f"Cylinder({outer_diam}, {height})"
        f".Sub(Cylinder({inner_diam}, {height}))"
    )

    params = {
        "R_outer_m": R_outer,
        "R_inner_m": R_inner,
        "wall_thickness_m": wall_nm * 1e-9,
        "length_m": L,
    }

    equation = (
        f"Outer surface: x² + y² = ({R_outer:.6e})², "
        f"-{L/2:.6e} ≤ z ≤ {L/2:.6e}; "
        f"Inner surface: x² + y² = ({R_inner:.6e})², "
        f"-{L/2:.6e} ≤ z ≤ {L/2:.6e}"
    )

    bbox = (2 * R_outer, 2 * R_outer, L)
    return shape_expr, params, equation, bbox, "nanotube"


def generate_torus(rng: random.Random):
    """
    Тор с большим радиусом R и малым радиусом r.

    Используется метод **Z-нарезки**, который точно ложится на конечно-разностную сетку:
    тор нарезается на горизонтальные слои толщиной = cell_size.
    На каждом уровне z неявное уравнение тора

        (sqrt(x² + y²) - R)² + z² = r²

    решается аналитически для получения кольцевого сечения:

        R_out(z) = R + sqrt(r² - z²)
        R_in(z)  = R - sqrt(r² - z²)

    Каждый слой становится  Cylinder(2·R_out, dz).Sub(Cylinder(2·R_in, dz)),
    сдвинутым на нужную позицию z
    """
    R_major_nm = rng.uniform(30, 80)
    r_minor_nm = rng.uniform(8, min(25, R_major_nm - 5))

    R  = R_major_nm * 1e-9          # большой радиус (м)
    r  = r_minor_nm * 1e-9          # малый радиус (м)
    dz = CELL_SIZE_M                # толщина среза = размер ячейки

    # Количество слоев z, охватывающих тор на интервале [-r, +r]
    n_slices = math.ceil(2 * r / dz)

    slices: list[str] = []
    for i in range(n_slices):
        z_lo     = -r + i * dz
        z_hi     = z_lo + dz
        z_center = 0.5 * (z_lo + z_hi)

        # Создаем срез только если его центр находится внутри тора
        if abs(z_center) >= r:
            continue

        # Аналитические радиусы кольца из уравнения тора
        delta = math.sqrt(r**2 - z_center**2)
        R_out = R + delta           # внешний радиус кольца на высоте z
        R_in  = R - delta           # внутренний радиус кольца на высоте z

        outer_d = f"{2 * R_out:.6e}"
        inner_d = f"{2 * R_in:.6e}"
        h       = f"{dz:.6e}"
        z_pos   = f"{z_center:.6e}"

        s = (
            f"Cylinder({outer_d}, {h})"
            f".Sub(Cylinder({inner_d}, {h}))"
            f".Transl(0, 0, {z_pos})"
        )
        slices.append(s)

    # Соединяем все z-срезы с помощью .Add()
    shape_expr = slices[0]
    for s in slices[1:]:
        shape_expr = f"{shape_expr}.Add({s})"

    params = {
        "R_major_m": R,
        "r_minor_m": r,
        "z_slices": len(slices),
        "slice_thickness_m": dz,
    }

    equation = (
        f"(sqrt(x² + y²) - {R:.6e})² + z² = ({r:.6e})²"
    )

    bbox = (2 * (R + r), 2 * (R + r), 2 * r)
    return shape_expr, params, equation, bbox, "torus"


def generate_hollow_sphere(rng: random.Random):
    """Полая сфера, построенная через вычитание эллипсоидов"""
    R_outer_nm = rng.uniform(20, 60)
    wall_nm    = rng.uniform(4, min(15, R_outer_nm - 4))

    R_outer = R_outer_nm * 1e-9
    R_inner = (R_outer_nm - wall_nm) * 1e-9

    outer_diam = f"{2*R_outer:.6e}"
    inner_diam = f"{2*R_inner:.6e}"

    shape_expr = (
        f"Ellipsoid({outer_diam}, {outer_diam}, {outer_diam})"
        f".Sub(Ellipsoid({inner_diam}, {inner_diam}, {inner_diam}))"
    )

    params = {
        "R_outer_m": R_outer,
        "R_inner_m": R_inner,
        "wall_thickness_m": wall_nm * 1e-9,
    }

    equation = (
        f"Outer surface: x² + y² + z² = ({R_outer:.6e})²; "
        f"Inner surface: x² + y² + z² = ({R_inner:.6e})²"
    )

    bbox = (2 * R_outer, 2 * R_outer, 2 * R_outer)
    return shape_expr, params, equation, bbox, "hollow_sphere"


def generate_chiral_twisted_wire(rng: random.Random):
    """
    Прямая сплошная закрученная нанонить (Chiral Twisted Wire).
    Генерируется прямой стержень с эллиптическим сечением, которое плавно закручивается вдоль оси Z.
    """
    # ── Случайная генерация параметров ──────────────────────────────
    D_major_nm = rng.uniform(40.0, 80.0)      # большой диаметр эллипса [нм]
    D_minor_nm = rng.uniform(15.0, 30.0)      # малый диаметр эллипса [нм]
    length_nm  = rng.uniform(100.0, 300.0)    # длина стержня [нм]
    
    n_turns    = rng.uniform(1.0, 4.0)        # количество полных оборотов сечения
    chirality  = rng.choice([-1, 1])          # хиральность: правое (+1) или левое (-1) закручивание
    
    # ── Перевод в СИ (метры) ───────────────────────────────────────
    D_major = D_major_nm * 1e-9
    D_minor = D_minor_nm * 1e-9
    L       = length_nm * 1e-9
    
    # ── Дискретизация по оси Z ─────────────────────────────────────
    # Шаг нарезки равен размеру ячейки симуляции
    dz = CELL_SIZE_M
    n_slices = max(2, math.ceil(L / dz))
    dz = L / n_slices  # точный равномерный шаг
    
    slices: list[str] = []
    
    scale_y = D_minor / D_major
    d_str = f"{D_major:.6e}"
    dz_str = f"{dz:.6e}"
    
    for i in range(n_slices):
        # Центр текущего слоя по Z
        z_center = -L / 2.0 + i * dz + dz / 2.0
        
        # Угол поворота сечения на данной высоте (от 0 до 2*pi*n_turns)
        theta = chirality * (2.0 * math.pi * n_turns) * (z_center + L / 2.0) / L
        
        # Строим слой: Цилиндр -> сплющиваем в эллипс -> поворачиваем -> сдвигаем по Z
        slice_expr = (
            f"Cylinder({d_str}, {dz_str})"
            f".Scale(1.0, {scale_y:.6e}, 1.0)"
            f".RotZ({theta:.6e})"
            f".Transl(0, 0, {z_center:.6e})"
        )
        slices.append(slice_expr)
        
    # ── Сборка сбалансированного бинарного CSG-дерева ──────────────
    def build_csg_tree(nodes: list[str]) -> str:
        if not nodes:
            return ""
        if len(nodes) == 1:
            return nodes[0]
        mid = len(nodes) // 2
        left = build_csg_tree(nodes[:mid])
        right = build_csg_tree(nodes[mid:])
        return f"({left}).Add({right})"
        
    shape_expr = build_csg_tree(slices)
    
    # ── Словарь параметров ────────────────────────────────────────
    params = {
        "D_major_m": D_major,
        "D_minor_m": D_minor,
        "length_m": L,
        "n_turns": n_turns,
        "chirality": chirality,
        "z_slices": n_slices,
        "slice_thickness_m": dz
    }
    
    # ── Аналитическое уравнение поверхности ────────────────────────
    chi_sym = "+1" if chirality == 1 else "-1"
    equation = (
        f"Twisted solid wire along Z: cross-section is an ellipse with axes "
        f"a={D_major/2:.6e}, b={D_minor/2:.6e}. "
        f"Rotates around Z by theta(z) = {chi_sym} * 2pi * {n_turns} * (z + L/2)/L, "
        f"for z in [-{L/2:.6e}, {L/2:.6e}]."
    )
    
    # ── Ограничивающий параллелепипед ─────────────────────────────
    # При вращении эллипс описывает цилиндр диаметром D_major
    bbox_x = D_major
    bbox_y = D_major
    bbox_z = L
    
    return shape_expr, params, equation, (bbox_x, bbox_y, bbox_z), "twisted_nanowire"

# ---------------------------------------------------------------------------
# Запись скрипта .mx3
# ---------------------------------------------------------------------------
MX3_TEMPLATE = textwrap.dedent("""\
    // ---------------------------------------------------------------
    // Автосгенерированный скрипт MuMax3  –  {shape_type}
    // ---------------------------------------------------------------

    // --- Сетка ---
    SetGridSize({Nx}, {Ny}, {Nz})
    SetCellSize(2e-9, 2e-9, 2e-9)

    // --- Сглаживание краев ---
    EdgeSmooth = 8

    // --- Геометрия ---
    SetGeom({shape_expr})

    // --- Параметры материала ---
    Msat  = {Msat}
    Aex   = {Aex}
    alpha = {alpha}

    // --- Формат вывода ---
    OutputFormat = OVF2_BINARY

    // --- Начальная намагниченность (случайная, с сидом) ---
    m = RandomMagSeed({seed})

    // --- Релаксация к минимуму энергии ---
    Relax()

    // --- Сохранение релаксированной намагниченности ---
    SaveAs(m, "m_relaxed")
""")


def write_mx3(
    path: Path,
    shape_type: str,
    shape_expr: str,
    Nx: int, Ny: int, Nz: int,
    seed: int,
):
    content = MX3_TEMPLATE.format(
        shape_type=shape_type,
        Nx=Nx, Ny=Ny, Nz=Nz,
        shape_expr=shape_expr,
        Msat=MSAT,
        Aex=AEX,
        alpha=ALPHA,
        seed=seed,
    )
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Запись geometry_info.json
# ---------------------------------------------------------------------------
def write_geometry_info(
    path: Path,
    shape_type: str,
    params: dict[str, Any],
    equation: str,
    Nx: int, Ny: int, Nz: int,
):
    info = {
        "shape_type": shape_type,
        "parameters": params,
        "grid": {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "cell_size_m": CELL_SIZE_M,
        },
        "surface_equation": equation,
        "material": {
            "Msat_A_per_m": MSAT,
            "Aex_J_per_m": AEX,
            "alpha": ALPHA,
        },
    }
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


# ===================================================================
# Главная точка входа
# ===================================================================
GENERATORS = [generate_nanotube, generate_torus, generate_hollow_sphere, generate_chiral_twisted_wire]


def main():
    parser = argparse.ArgumentParser(
        description="Генерация папок с симуляциями MuMax3 для обучения GNN."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Количество генерируемых конфигураций (по умолчанию: 100)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset",
        help="Корневая директория для вывода (по умолчанию: 'dataset')",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Главный случайный сид для воспроизводимости (по умолчанию: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task_lines: list[str] = []

    for idx in range(args.num_samples):
        sim_name = f"sim_{idx:04d}"
        sim_dir  = output_root / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        # Выбираем случайный тип геометрии
        gen_func = rng.choice(GENERATORS)
        shape_expr, params, equation, bbox, shape_type = gen_func(rng)

        # Вычисляем сетку, удобную для БПФ
        Nx, Ny, Nz = compute_grid(*bbox)

        # Уникальный сид для RandomMagSeed внутри MuMax3
        mag_seed = rng.randint(1, 2**31 - 1)

        # Записываем .mx3
        write_mx3(
            sim_dir / "sim.mx3",
            shape_type=shape_type,
            shape_expr=shape_expr,
            Nx=Nx, Ny=Ny, Nz=Nz,
            seed=mag_seed,
        )

        # Записываем geometry_info.json
        write_geometry_info(
            sim_dir / "geometry_info.json",
            shape_type=shape_type,
            params=params,
            equation=equation,
            Nx=Nx, Ny=Ny, Nz=Nz,
        )

        task_lines.append(str(sim_dir))

        print(f"[{idx+1:4d}/{args.num_samples}]  {sim_name}  ({shape_type})"
              f"  grid=({Nx},{Ny},{Nz})")

    # Записываем tasks.txt для SLURM
    tasks_path = output_root / "tasks.txt"
    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")

    print(f"\nГотово. {args.num_samples} симуляций записано в {output_root}")
    print(f"Список задач для SLURM: {tasks_path}")


if __name__ == "__main__":
    main()
