# env_v2.py
# Road Inspection RL Environment — verzija 2
# Novo: MINOR_DAMAGE (tip 2) i MAJOR_DAMAGE (tip 3) — različite kazne i bonusi
# Grid: 10x10, gušći raspored prepreka i oštećenja
# Cilj: donji desni ugao (9,9)

import numpy as np
import tkinter as tk
import time

# ── Dimenzije mreže ──────────────────────────────────────────────
PIXELS      = 55   # malo manji piksel jer je 10x10
GRID_H      = 10
GRID_W      = 10

# ── Tipovi ćelija ────────────────────────────────────────────────
FREE         = 0   # slobodna cesta          → reward -0.05
ROAD_CLOSED  = 1   # zatvorena (neprohodna)  → reward -1.0, done=True
MINOR_DAMAGE = 2   # manje oštećenje         → reward -0.3 (+0.2 bonus za novo)
MAJOR_DAMAGE = 3   # veće oštećenje          → reward -0.7 (+0.5 bonus za novo)
GOAL         = 4   # cilj                    → reward +1.0, done=True

# ── Grid 10x10 ───────────────────────────────────────────────────
# 0=slobodno  1=zatvoreno  2=minor  3=major  4=cilj
# Start: (0,0) gornji lijevi,  Cilj: (9,9) donji desni
# Osigurano da postoji barem jedan prohodan put
# Gustoća oštećenih/zatvorenih ćelija: ~42%
GRID_MAP = np.array([
    [0, 2, 0, 1, 0, 3, 0, 2, 0, 0],
    [0, 0, 1, 0, 2, 0, 1, 0, 3, 0],
    [3, 0, 0, 2, 0, 0, 0, 1, 0, 2],
    [0, 1, 2, 0, 1, 3, 0, 0, 2, 0],
    [0, 0, 0, 1, 0, 0, 2, 0, 1, 3],
    [2, 3, 0, 0, 2, 1, 0, 3, 0, 0],
    [0, 0, 1, 3, 0, 0, 1, 0, 2, 0],
    [1, 2, 0, 0, 3, 0, 0, 2, 0, 1],
    [0, 0, 3, 1, 0, 2, 0, 0, 1, 0],
    [0, 2, 0, 0, 1, 0, 3, 0, 2, 4],
], dtype=int)

# ── Boje ćelija (trening grid) ────────────────────────────────────
CELL_COLORS = {
    FREE:         '#F1EFE8',   # svijetlo siva  — slobodna cesta
    ROAD_CLOSED:  '#E24B4A',   # crvena         — zatvorena
    MINOR_DAMAGE: '#EF9F27',   # narančasta      — manje oštećenje
    MAJOR_DAMAGE: '#C0392B',   # tamno crvena    — veće oštećenje
    GOAL:         '#1D9E75',   # zelena          — cilj
}
AGENT_COLOR         = '#185FA5'   # plava         — agent
VISITED_COLOR       = '#B5D4F4'   # svijetlo plava — posjećena slobodna ćelija
MINOR_FOUND_COLOR   = '#FAC775'   # žuta           — pronađeni minor damage
MAJOR_FOUND_COLOR   = '#E8735A'   # koraljna       — pronađeni major damage


class Environment(tk.Tk, object):

    def __init__(self, render_speed=0.05):
        super().__init__()
        self.title('Road Inspection v2 — Q-Learning  |  10×10  |  Minor + Major Damage')
        self.geometry(f'{GRID_W * PIXELS}x{GRID_H * PIXELS + 65}')
        self.resizable(False, False)

        self.action_space  = ['gore', 'dolje', 'desno', 'lijevo']
        self.n_actions     = len(self.action_space)
        self.render_speed  = render_speed

        # Statistike
        self.shortest_route   = None
        self.shortest_steps   = float('inf')
        self.damage_found_log = []   # sve pronađene damage ćelije (minor + major)
        self.episode_count    = 0
        self.success_count    = 0

        self._build_canvas()
        self._draw_grid()

    # ── Izgradnja platna ──────────────────────────────────────────
    def _build_canvas(self):
        self.canvas = tk.Canvas(
            self, bg='#1E1E2E',
            width=GRID_W * PIXELS, height=GRID_H * PIXELS
        )
        self.canvas.pack()

        self.info_var = tk.StringVar(value='Epizoda: 0  |  Koraci: —  |  Oštećenja: 0')
        tk.Label(
            self, textvariable=self.info_var,
            font=('Courier', 9), bg='#2A2A3E', fg='#E8E6DF'
        ).pack(fill='x')

    # ── Crtanje mreže ─────────────────────────────────────────────
    def _draw_grid(self):
        self.cell_rects = {}

        for r in range(GRID_H):
            for c in range(GRID_W):
                x0 = c * PIXELS
                y0 = r * PIXELS
                x1 = x0 + PIXELS
                y1 = y0 + PIXELS
                cell_type = GRID_MAP[r, c]
                color     = CELL_COLORS[cell_type]

                rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color, outline='#3A3A50', width=1
                )
                self.cell_rects[(r, c)] = rect_id

                # Oznake
                label_map = {
                    ROAD_CLOSED:  ('✕', '#FFFFFF'),
                    MINOR_DAMAGE: ('m', '#2C2C2A'),
                    MAJOR_DAMAGE: ('M', '#FFFFFF'),
                    GOAL:         ('G', '#FFFFFF'),
                    FREE:         ('',  ''),
                }
                lbl, lbl_color = label_map.get(cell_type, ('', ''))
                if lbl:
                    self.canvas.create_text(
                        x0 + PIXELS // 2, y0 + PIXELS // 2,
                        text=lbl, font=('Courier', 12, 'bold'),
                        fill=lbl_color
                    )

        # Agent
        self._agent_pos  = [0, 0]
        self.agent_shape = self.canvas.create_oval(
            *self._agent_bbox(0, 0), fill=AGENT_COLOR, outline='#042C53', width=2
        )

    def _agent_bbox(self, row, col, margin=9):
        x0 = col * PIXELS + margin
        y0 = row * PIXELS + margin
        return x0, y0, x0 + PIXELS - 2 * margin, y0 + PIXELS - 2 * margin

    # ── Reset ─────────────────────────────────────────────────────
    def reset(self):
        self.update()
        self.episode_count += 1
        self._step_count          = 0
        self._damage_this_episode = []
        self._visited             = set()
        self._current_path        = [(0, 0)]

        # Vraćamo originalne boje posjećenih ćelija
        for (r, c) in getattr(self, '_visited_last', set()):
            ct = GRID_MAP[r, c]
            if ct in (MINOR_DAMAGE, MAJOR_DAMAGE):
                pass   # damage ćelije ostaju obojene (pronađene)
            else:
                self.canvas.itemconfig(self.cell_rects[(r, c)], fill=CELL_COLORS[ct])

        self._visited_last = set()
        self._agent_pos    = [0, 0]
        self.canvas.coords(self.agent_shape, *self._agent_bbox(0, 0))
        self.canvas.lift(self.agent_shape)

        return self._state_key()

    # ── Korak agenta ──────────────────────────────────────────────
    def step(self, action):
        r, c = self._agent_pos

        dr, dc = 0, 0
        if   action == 0 and r > 0:          dr = -1
        elif action == 1 and r < GRID_H - 1: dr =  1
        elif action == 2 and c < GRID_W - 1: dc =  1
        elif action == 3 and c > 0:          dc = -1

        nr, nc    = r + dr, c + dc
        cell_type = GRID_MAP[nr, nc]

        # Bojanje posjećene slobodne ćelije
        if GRID_MAP[r, c] == FREE and (r, c) != (0, 0):
            self.canvas.itemconfig(self.cell_rects[(r, c)], fill=VISITED_COLOR)
        self._visited.add((r, c))
        self._visited_last.add((r, c))

        # Pomak agenta
        self._agent_pos = [nr, nc]
        self.canvas.coords(self.agent_shape, *self._agent_bbox(nr, nc))
        self.canvas.lift(self.agent_shape)
        self._step_count += 1
        self._current_path.append((nr, nc))

        # ── Reward logika ─────────────────────────────────────────
        if cell_type == GOAL:
            reward = 100.0
            done   = True
            self.success_count += 1
            if self._step_count < self.shortest_steps:
                self.shortest_steps = self._step_count
                self.shortest_route = list(self._current_path)

        elif cell_type == ROAD_CLOSED:
            reward = -1.0
            done   = True

        elif cell_type == MINOR_DAMAGE:
            reward = -0.3
            done   = False
            if (nr, nc) not in self._damage_this_episode:
                reward += 0.2   # bonus za inspekciju novog minor damage
                self._damage_this_episode.append((nr, nc))
                self.damage_found_log.append({
                    'episode': self.episode_count,
                    'type'   : 'minor',
                    'row': nr, 'col': nc,
                })
                self.canvas.itemconfig(self.cell_rects[(nr, nc)], fill=MINOR_FOUND_COLOR)

        elif cell_type == MAJOR_DAMAGE:
            reward = -0.7
            done   = False
            if (nr, nc) not in self._damage_this_episode:
                reward += 0.5   # veći bonus za inspekciju novog major damage
                self._damage_this_episode.append((nr, nc))
                self.damage_found_log.append({
                    'episode': self.episode_count,
                    'type'   : 'major',
                    'row': nr, 'col': nc,
                })
                self.canvas.itemconfig(self.cell_rects[(nr, nc)], fill=MAJOR_FOUND_COLOR)

        else:
            reward = -0.05
            done   = False

        # Info traka
        minor_found = sum(1 for d in self.damage_found_log if d['type'] == 'minor')
        major_found = sum(1 for d in self.damage_found_log if d['type'] == 'major')
        self.info_var.set(
            f'Ep: {self.episode_count}  |  '
            f'Koraci: {self._step_count}  |  '
            f'Najkraća: {self.shortest_steps if self.shortest_steps < float("inf") else "—"}  |  '
            f'Minor: {minor_found}  Major: {major_found}'
        )

        return self._state_key(), reward, done

    def _state_key(self):
        r, c = self._agent_pos
        return f'{r},{c}'

    def render(self):
        if self.render_speed > 0:
            time.sleep(self.render_speed)
        self.update()

    # ── Vizualni prikaz optimalne rute u novom prozoru ────────────
    def show_optimal_route_window(self):
        """Toplevel prozor s grid prikazom optimalne rute — v2 (minor + major damage)."""

        PX = 58   # ćelija u route prozoru

        win = tk.Toplevel(self)
        win.title('Optimalna ruta — v2  |  10×10')
        win.resizable(False, False)
        win.configure(bg='#1E1E2E')

        # ── Naslov ───────────────────────────────────────────────
        header = tk.Frame(win, bg='#1E1E2E')
        header.pack(fill='x', padx=12, pady=(12, 4))

        tk.Label(header, text='OPTIMALNA RUTA',
                 font=('Courier', 13, 'bold'),
                 bg='#1E1E2E', fg='#E8E6DF').pack(side='left')

        steps_txt = f'{self.shortest_steps} koraka' if self.shortest_steps < float('inf') else '—'
        tk.Label(header, text=f'  ({steps_txt})',
                 font=('Courier', 11),
                 bg='#1E1E2E', fg='#7F77DD').pack(side='left')

        # ── Canvas ────────────────────────────────────────────────
        canvas = tk.Canvas(
            win, bg='#1E1E2E',
            width=GRID_W * PX, height=GRID_H * PX,
            highlightthickness=0
        )
        canvas.pack(padx=12, pady=4)

        # Boje route prozora
        C = {
            'route'        : '#7F77DD',   # ljubičasta — ruta (slobodne ćelije)
            'start'        : '#185FA5',   # plava      — start
            'goal'         : '#1D9E75',   # zelena     — cilj
            'minor_route'  : '#EF9F27',   # narančasta — minor na ruti
            'major_route'  : '#E24B4A',   # crvena     — major na ruti
            'closed'       : '#8B1A1A',   # tamno crvena — zatvorena
            'minor_off'    : '#3D2800',   # tamna narančasta — minor van rute
            'major_off'    : '#3D0A0A',   # tamno crvena — major van rute
            'free_off'     : '#2A2A3E',   # tamna — slobodna van rute
            'grid_line'    : '#3A3A50',
            'text_light'   : '#E8E6DF',
            'text_dark'    : '#1E1E2E',
        }

        route_set      = set(map(tuple, self.shortest_route)) if self.shortest_route else set()
        route_step_map = {}
        if self.shortest_route:
            for idx, pos in enumerate(self.shortest_route):
                route_step_map[tuple(pos)] = idx

        for r in range(GRID_H):
            for c in range(GRID_W):
                x0 = c * PX;  y0 = r * PX
                x1 = x0 + PX; y1 = y0 + PX
                pos       = (r, c)
                ct        = GRID_MAP[r, c]
                on_route  = pos in route_set

                # Boja ćelije
                if pos == (0, 0):
                    fill = C['start']
                elif ct == GOAL:
                    fill = C['goal']
                elif ct == ROAD_CLOSED:
                    fill = C['closed']
                elif on_route and ct == MINOR_DAMAGE:
                    fill = C['minor_route']
                elif on_route and ct == MAJOR_DAMAGE:
                    fill = C['major_route']
                elif on_route:
                    fill = C['route']
                elif ct == MINOR_DAMAGE:
                    fill = C['minor_off']
                elif ct == MAJOR_DAMAGE:
                    fill = C['major_off']
                else:
                    fill = C['free_off']

                canvas.create_rectangle(x0, y0, x1, y1,
                                        fill=fill, outline=C['grid_line'], width=1)

                cx = x0 + PX // 2
                cy = y0 + PX // 2

                # Tekst u ćeliji
                if pos == (0, 0):
                    canvas.create_text(cx, cy - 8, text='▶',
                                       font=('Courier', 13, 'bold'), fill=C['text_light'])
                    canvas.create_text(cx, cy + 9, text='START',
                                       font=('Courier', 6, 'bold'), fill=C['text_light'])

                elif ct == GOAL:
                    canvas.create_text(cx, cy - 8, text='★',
                                       font=('Courier', 13, 'bold'), fill=C['text_light'])
                    canvas.create_text(cx, cy + 9, text='CILJ',
                                       font=('Courier', 6, 'bold'), fill=C['text_light'])

                elif ct == ROAD_CLOSED:
                    canvas.create_text(cx, cy, text='✕',
                                       font=('Courier', 14, 'bold'), fill=C['text_light'])

                elif on_route:
                    step_num = route_step_map.get(pos, '')
                    canvas.create_text(cx, cy - 6, text=str(step_num),
                                       font=('Courier', 10, 'bold'), fill=C['text_light'])
                    # Oznaka tipa oštećenja na ruti
                    if ct == MINOR_DAMAGE:
                        canvas.create_text(cx, cy + 9, text='m',
                                           font=('Courier', 8, 'bold'), fill=C['text_dark'])
                    elif ct == MAJOR_DAMAGE:
                        canvas.create_text(cx, cy + 9, text='M',
                                           font=('Courier', 8, 'bold'), fill=C['text_dark'])

                elif ct == MINOR_DAMAGE:
                    canvas.create_text(cx, cy, text='m',
                                       font=('Courier', 11, 'bold'), fill='#EF9F27')
                elif ct == MAJOR_DAMAGE:
                    canvas.create_text(cx, cy, text='M',
                                       font=('Courier', 11, 'bold'), fill='#E24B4A')

        # ── Strelice ─────────────────────────────────────────────
        if self.shortest_route and len(self.shortest_route) > 1:
            for i in range(len(self.shortest_route) - 1):
                r0, c0 = self.shortest_route[i]
                r1, c1 = self.shortest_route[i + 1]
                sx = c0 * PX + PX // 2;  sy = r0 * PX + PX // 2
                ex = c1 * PX + PX // 2;  ey = r1 * PX + PX // 2
                dx = ex - sx;            dy = ey - sy
                canvas.create_line(
                    sx + dx * 0.35, sy + dy * 0.35,
                    ex - dx * 0.35, ey - dy * 0.35,
                    arrow=tk.LAST, fill='#FFFFFF',
                    width=2, arrowshape=(7, 9, 3)
                )

        # ── Legenda ───────────────────────────────────────────────
        legend_frame = tk.Frame(win, bg='#1E1E2E')
        legend_frame.pack(padx=12, pady=(4, 8), fill='x')

        legend_items = [
            (C['start'],        'Start (0,0)'),
            (C['route'],        'Ruta'),
            (C['goal'],         'Cilj'),
            (C['minor_route'],  'Minor na ruti  (m)'),
            (C['major_route'],  'Major na ruti  (M)'),
            (C['closed'],       'Zatvorena cesta'),
            (C['minor_off'],    'Minor (van rute)'),
            (C['major_off'],    'Major (van rute)'),
        ]

        for i, (color, label) in enumerate(legend_items):
            cf = tk.Frame(legend_frame, bg='#1E1E2E')
            cf.grid(row=i // 4, column=i % 4, padx=6, pady=2, sticky='w')
            tk.Label(cf, bg=color, width=2, height=1).pack(side='left', padx=(0, 3))
            tk.Label(cf, text=label, font=('Courier', 7),
                     bg='#1E1E2E', fg='#B4B2A9').pack(side='left')

        # ── Statistike oštećenja ──────────────────────────────────
        stats_frame = tk.Frame(win, bg='#2A2A3E')
        stats_frame.pack(padx=12, pady=(0, 6), fill='x')

        total_minor = int(np.sum(GRID_MAP == MINOR_DAMAGE))
        total_major = int(np.sum(GRID_MAP == MAJOR_DAMAGE))
        found_minor = sum(1 for d in self.damage_found_log if d['type'] == 'minor')
        found_major = sum(1 for d in self.damage_found_log if d['type'] == 'major')

        stats_txt = (
            f'Minor pronađeno: {found_minor}/{total_minor}  '
            f'({100*found_minor//max(total_minor,1)}%)     '
            f'Major pronađeno: {found_major}/{total_major}  '
            f'({100*found_major//max(total_major,1)}%)'
        )
        tk.Label(stats_frame, text=stats_txt,
                 font=('Courier', 8), bg='#2A2A3E', fg='#E8E6DF').pack(pady=4)

        # ── Gumb zatvori ──────────────────────────────────────────
        tk.Button(
            win, text='  Zatvori  ',
            font=('Courier', 9, 'bold'),
            bg='#7F77DD', fg='white',
            activebackground='#5C55BB',
            relief='flat', cursor='hand2',
            command=win.destroy
        ).pack(pady=(0, 10))

        # Pozicija pored glavnog prozora
        x = self.winfo_x() + self.winfo_width() + 10
        y = self.winfo_y()
        win.geometry(f'+{x}+{y}')
        win.lift()
        win.focus_force()

    # ── Finalni ispis ─────────────────────────────────────────────
    def show_final_route(self):
        print('\n═══ REZULTATI v2 ═══')
        print(f'Uspješnih epizoda  : {self.success_count} / {self.episode_count}')
        print(f'Najkraća ruta      : {self.shortest_steps} koraka')
        total_minor = int(np.sum(GRID_MAP == MINOR_DAMAGE))
        total_major = int(np.sum(GRID_MAP == MAJOR_DAMAGE))
        found_minor = sum(1 for d in self.damage_found_log if d['type'] == 'minor')
        found_major = sum(1 for d in self.damage_found_log if d['type'] == 'major')
        print(f'Minor pronađeno    : {found_minor} / {total_minor}')
        print(f'Major pronađeno    : {found_major} / {total_major}')

    def get_stats(self):
        return {
            'total_episodes' : self.episode_count,
            'success_count'  : self.success_count,
            'shortest_steps' : self.shortest_steps if self.shortest_steps < float('inf') else None,
            'damage_log'     : self.damage_found_log,
            'minor_found'    : sum(1 for d in self.damage_found_log if d['type'] == 'minor'),
            'major_found'    : sum(1 for d in self.damage_found_log if d['type'] == 'major'),
        }
