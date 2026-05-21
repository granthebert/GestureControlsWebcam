import sys
from dataclasses import dataclass
from enum import Enum, auto

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Signal,
    QObject,
)
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)


class GestureAction(Enum):
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    LOCK_PAGE = auto()
    UNLOCK_PAGE = auto()
    SELECT = auto()
    BACK = auto()


class GestureEventBus(QObject):
    action_received = Signal(GestureAction)

    def emit_action(self, action: GestureAction) -> None:
        self.action_received.emit(action)


@dataclass(frozen=True)
class ActionTile:
    icon: str
    title: str
    subtitle: str


class FocusTile(QFrame):
    activated = Signal(str)

    def __init__(self, tile: ActionTile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tile = tile
        self._focused = False

        self.setObjectName("focusTile")
        self.setProperty("focused", "false")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(220, 150)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 10)
        self.shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(self.shadow)

        icon_label = QLabel(tile.icon)
        icon_label.setObjectName("tileIcon")
        icon_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(tile.title)
        title_label.setObjectName("tileTitle")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel(tile.subtitle)
        subtitle_label.setObjectName("tileSubtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(8)
        layout.addStretch(1)
        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addStretch(1)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.activate()

    def activate(self) -> None:
        self.activated.emit(self.tile.title)
        self._pulse()

    def set_focused(self, focused: bool) -> None:
        if self._focused == focused:
            return

        self._focused = focused
        self.setProperty("focused", "true" if focused else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

        self.shadow.setBlurRadius(48 if focused else 20)
        self.shadow.setOffset(0, 0 if focused else 10)
        self.shadow.setColor(
            QColor(78, 220, 255, 190) if focused else QColor(0, 0, 0, 100)
        )

    def _pulse(self) -> None:
        self.shadow.setColor(QColor(255, 255, 255, 230))
        QTimer.singleShot(
            120,
            lambda: self.shadow.setColor(
                QColor(78, 220, 255, 190)
                if self._focused
                else QColor(0, 0, 0, 100)
            ),
        )


class FocusManager(QObject):
    focus_changed = Signal(int)

    def __init__(self, items: list[FocusTile], columns: int = 3) -> None:
        super().__init__()
        self.items = items
        self.index = 0
        self.columns = max(1, columns)

    def set_enabled(self, enabled: bool) -> None:
        if not self.items:
            return
        for item in self.items:
            item.set_focused(False)
        if enabled:
            self.items[self.index].set_focused(True)

    def move_next(self) -> None:
        self._move(1)

    def move_previous(self) -> None:
        self._move(-1)

    def move_up(self) -> None:
        self._move_vertical(-1)

    def move_down(self) -> None:
        self._move_vertical(1)

    def select_current(self) -> None:
        if self.items:
            self.items[self.index].activate()

    def _move(self, delta: int) -> None:
        if not self.items:
            return
        self.items[self.index].set_focused(False)
        self.index = (self.index + delta) % len(self.items)
        self.items[self.index].set_focused(True)
        self.focus_changed.emit(self.index)

    def _move_vertical(self, direction: int) -> None:
        if not self.items:
            return

        row = self.index // self.columns
        column = self.index % self.columns
        rows = (len(self.items) + self.columns - 1) // self.columns
        target_row = (row + direction) % rows
        target = target_row * self.columns + column

        while target >= len(self.items):
            target -= self.columns

        if target == self.index:
            return

        self.items[self.index].set_focused(False)
        self.index = target
        self.items[self.index].set_focused(True)
        self.focus_changed.emit(self.index)


class DashboardPage(QWidget):
    item_selected = Signal(str, str)

    def __init__(
        self,
        name: str,
        tagline: str,
        accent: str,
        tiles: list[ActionTile],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.accent = accent
        self.setObjectName("dashboardPage")

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(72, 52, 72, 52)
        page_layout.setSpacing(28)

        header = QHBoxLayout()
        header.setSpacing(18)

        title_block = QVBoxLayout()
        title_block.setSpacing(6)

        title = QLabel(name)
        title.setObjectName("pageTitle")

        subtitle = QLabel(tagline)
        subtitle.setObjectName("pageSubtitle")

        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        accent_dot = QLabel()
        accent_dot.setFixedSize(16, 64)
        accent_dot.setStyleSheet(
            f"background: {accent}; border-radius: 8px;"
        )

        header.addWidget(accent_dot)
        header.addLayout(title_block)
        header.addStretch(1)

        self.status_label = QLabel("Unlocked: swipe changes pages")
        self.status_label.setObjectName("pageStatus")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self.status_label)

        grid = QGridLayout()
        grid.setSpacing(22)
        self.tiles: list[FocusTile] = []
        for index, tile in enumerate(tiles):
            focus_tile = FocusTile(tile)
            focus_tile.activated.connect(self._on_tile_selected)
            self.tiles.append(focus_tile)
            grid.addWidget(focus_tile, index // 3, index % 3)

        page_layout.addLayout(header)
        page_layout.addLayout(grid, 1)

        self.focus_manager = FocusManager(self.tiles, columns=3)

    def set_locked(self, locked: bool) -> None:
        self.status_label.setText(
            "Locked: swipe moves focus" if locked else "Unlocked: swipe changes pages"
        )
        self.status_label.setProperty("locked", locked)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.focus_manager.set_enabled(locked)

    def _on_tile_selected(self, title: str) -> None:
        self.item_selected.emit(self.name, title)


class PageManager(QWidget):
    page_changed = Signal(int, str)
    page_locked_changed = Signal(bool)

    def __init__(self, pages: list[DashboardPage], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pages = pages
        self.current_index = 0
        self.locked = False
        self._animating = False

        self.viewport = QWidget(self)
        self.viewport.setObjectName("pageViewport")
        self.stack_layout = QStackedLayout(self.viewport)
        self.stack_layout.setStackingMode(QStackedLayout.StackAll)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        for page in pages:
            self.stack_layout.addWidget(page)
            page.hide()

        self.pages[0].show()
        self.pages[0].set_locked(False)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.viewport.setGeometry(self.rect())
        for page in self.pages:
            page.setGeometry(self.viewport.rect())
        super().resizeEvent(event)

    def handle_action(self, action: GestureAction) -> None:
        if action == GestureAction.LOCK_PAGE:
            self.set_locked(True)
        elif action in (GestureAction.UNLOCK_PAGE, GestureAction.BACK):
            self.set_locked(False)
        elif action == GestureAction.SELECT and self.locked:
            self.current_page.focus_manager.select_current()
        elif action == GestureAction.SWIPE_LEFT:
            self._handle_swipe(1)
        elif action == GestureAction.SWIPE_RIGHT:
            self._handle_swipe(-1)
        elif action == GestureAction.SWIPE_UP and self.locked:
            self.current_page.focus_manager.move_up()
        elif action == GestureAction.SWIPE_DOWN and self.locked:
            self.current_page.focus_manager.move_down()

    @property
    def current_page(self) -> DashboardPage:
        return self.pages[self.current_index]

    def set_locked(self, locked: bool) -> None:
        if self.locked == locked:
            return
        self.locked = locked
        self.current_page.set_locked(locked)
        self.page_locked_changed.emit(locked)

    def _handle_swipe(self, delta: int) -> None:
        if self.locked:
            if delta > 0:
                self.current_page.focus_manager.move_next()
            else:
                self.current_page.focus_manager.move_previous()
            return

        self.slide_to(self.current_index + delta)

    def slide_to(self, target_index: int) -> None:
        if self._animating or target_index == self.current_index:
            return

        if not 0 <= target_index < len(self.pages):
            self._edge_bump(-1 if target_index < 0 else 1)
            return

        self._animating = True
        direction = 1 if target_index > self.current_index else -1
        width = max(1, self.viewport.width())

        current_page = self.pages[self.current_index]
        next_page = self.pages[target_index]

        current_page.set_locked(False)
        next_page.set_locked(False)
        next_page.setGeometry(QRect(direction * width, 0, width, self.viewport.height()))
        next_page.show()
        next_page.raise_()

        outgoing = QPropertyAnimation(current_page, b"pos", self)
        outgoing.setDuration(360)
        outgoing.setStartValue(QPoint(0, 0))
        outgoing.setEndValue(QPoint(-direction * width, 0))
        outgoing.setEasingCurve(QEasingCurve.OutCubic)

        incoming = QPropertyAnimation(next_page, b"pos", self)
        incoming.setDuration(360)
        incoming.setStartValue(QPoint(direction * width, 0))
        incoming.setEndValue(QPoint(0, 0))
        incoming.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(outgoing)
        group.addAnimation(incoming)
        group.finished.connect(lambda: self._finish_slide(current_page, target_index, group))
        group.start(QAbstractAnimation.KeepWhenStopped)

    def _finish_slide(
        self, previous_page: DashboardPage, target_index: int, group: QParallelAnimationGroup
    ) -> None:
        previous_page.hide()
        previous_page.move(0, 0)
        self.current_index = target_index
        self.pages[target_index].setGeometry(self.viewport.rect())
        self._animating = False
        self.page_changed.emit(target_index, self.pages[target_index].name)
        group.deleteLater()

    def _edge_bump(self, direction: int) -> None:
        if self._animating:
            return

        self._animating = True
        page = self.current_page
        distance = 42 * direction

        out = QPropertyAnimation(page, b"pos", self)
        out.setDuration(110)
        out.setStartValue(QPoint(0, 0))
        out.setEndValue(QPoint(-distance, 0))
        out.setEasingCurve(QEasingCurve.OutQuad)

        back = QPropertyAnimation(page, b"pos", self)
        back.setDuration(160)
        back.setStartValue(QPoint(-distance, 0))
        back.setEndValue(QPoint(0, 0))
        back.setEasingCurve(QEasingCurve.OutBack)

        out.finished.connect(back.start)
        back.finished.connect(lambda: setattr(self, "_animating", False))
        out.start(QAbstractAnimation.DeleteWhenStopped)


class GestureDashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gesture Dashboard")
        self.resize(1280, 760)
        self.setMinimumSize(960, 620)
        self.control_surface_active = True

        self.event_bus = GestureEventBus()
        self.event_bus.action_received.connect(self._handle_gesture_action)

        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(14)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(16)

        brand = QLabel("GesturePY")
        brand.setObjectName("brand")

        self.mode_label = QLabel("PAGE NAVIGATION")
        self.mode_label.setObjectName("modePill")
        self.mode_label.setAlignment(Qt.AlignCenter)

        self.page_indicator = QLabel()
        self.page_indicator.setObjectName("pageIndicator")
        self.page_indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addWidget(brand)
        top_bar.addWidget(self.mode_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.page_indicator)

        self.toast = QLabel("")
        self.toast.setObjectName("toast")
        self.toast.setAlignment(Qt.AlignCenter)
        self.toast.hide()

        self.page_manager = PageManager(self._create_pages())
        self.page_manager.page_changed.connect(self._on_page_changed)
        self.page_manager.page_locked_changed.connect(self._on_lock_changed)

        layout.addLayout(top_bar)
        layout.addWidget(self.page_manager, 1)
        layout.addWidget(self.toast)

        self._apply_style()
        self._on_page_changed(0, self.page_manager.current_page.name)

    def dispatch_gesture(self, action: GestureAction) -> None:
        """Public integration point for camera or gesture recognition code."""
        if not self.control_surface_active:
            return
        self.event_bus.emit_action(action)

    def set_control_surface_active(self, active: bool) -> None:
        self.control_surface_active = active
        if not active:
            self.page_manager.set_locked(False)
        self.mode_label.setText(
            "PAGE NAVIGATION" if active else "DESKTOP GESTURES"
        )
        self.mode_label.setProperty("locked", self.page_manager.locked and active)
        self.mode_label.style().unpolish(self.mode_label)
        self.mode_label.style().polish(self.mode_label)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        key_map = {
            Qt.Key_Left: GestureAction.SWIPE_RIGHT,
            Qt.Key_Right: GestureAction.SWIPE_LEFT,
            Qt.Key_Up: GestureAction.SWIPE_UP,
            Qt.Key_Down: GestureAction.SWIPE_DOWN,
            Qt.Key_Space: (
                GestureAction.UNLOCK_PAGE
                if self.page_manager.locked
                else GestureAction.LOCK_PAGE
            ),
            Qt.Key_Return: GestureAction.SELECT,
            Qt.Key_Enter: GestureAction.SELECT,
            Qt.Key_Escape: GestureAction.BACK,
            Qt.Key_Backspace: GestureAction.BACK,
        }

        action = key_map.get(event.key())
        if action:
            self.dispatch_gesture(action)
            return
        super().keyPressEvent(event)

    def _create_pages(self) -> list[DashboardPage]:
        pages = [
            DashboardPage(
                "Home",
                "Quick launch and system actions",
                "#4edcff",
                [
                    ActionTile("GO", "Launch", "Open the primary experience"),
                    ActionTile("REC", "Capture", "Start a gesture session"),
                    ActionTile("LIB", "Library", "Browse pinned actions"),
                    ActionTile("AI", "Assistant", "Show contextual commands"),
                    ActionTile("WIN", "Overview", "See active windows"),
                    ActionTile("PWR", "Sleep", "Dim the dashboard"),
                ],
            ),
            DashboardPage(
                "Media",
                "Distance-friendly playback controls",
                "#74ff9a",
                [
                    ActionTile(">", "Play", "Resume current media"),
                    ActionTile("||", "Pause", "Pause playback"),
                    ActionTile(">>", "Skip", "Next track or chapter"),
                    ActionTile("<<", "Replay", "Previous track or rewind"),
                    ActionTile("+", "Volume Up", "Increase output volume"),
                    ActionTile("-", "Volume Down", "Decrease output volume"),
                ],
            ),
            DashboardPage(
                "Browser",
                "Large target web navigation",
                "#ffce5c",
                [
                    ActionTile("TAB", "New Tab", "Open a fresh tab"),
                    ActionTile("RLD", "Refresh", "Reload current page"),
                    ActionTile("<", "Back", "Go to previous page"),
                    ActionTile(">", "Forward", "Go to next page"),
                    ActionTile("FND", "Search", "Focus the address bar"),
                    ActionTile("FAV", "Bookmark", "Save current page"),
                ],
            ),
            DashboardPage(
                "Gaming",
                "Console-style launcher controls",
                "#ff6f9c",
                [
                    ActionTile("RUN", "Resume", "Return to active game"),
                    ActionTile("LIB", "Library", "Browse installed games"),
                    ActionTile("MIC", "Voice", "Open party controls"),
                    ActionTile("REC", "Record", "Capture the last moment"),
                    ActionTile("CFG", "Profiles", "Switch control profile"),
                    ActionTile("X", "Quit", "Close active game"),
                ],
            ),
            DashboardPage(
                "Settings",
                "Gesture system tuning",
                "#b28cff",
                [
                    ActionTile("SEN", "Sensitivity", "Adjust gesture threshold"),
                    ActionTile("CLK", "Cooldown", "Tune repeat delay"),
                    ActionTile("HUD", "Overlay", "Toggle hand HUD"),
                    ActionTile("SUN", "Theme", "Cycle visual style"),
                    ActionTile("?", "Help", "Show controls"),
                    ActionTile("OK", "Calibrate", "Run hand calibration"),
                ],
            ),
        ]

        for page in pages:
            page.item_selected.connect(self._on_item_selected)
        return pages

    def _handle_gesture_action(self, action: GestureAction) -> None:
        self.page_manager.handle_action(action)

    def _on_page_changed(self, index: int, name: str) -> None:
        total = len(self.page_manager.pages)
        dots = ["[*]" if i == index else "[ ]" for i in range(total)]
        self.page_indicator.setText(f"{name}  {' '.join(dots)}")

    def _on_lock_changed(self, locked: bool) -> None:
        self.mode_label.setText("FOCUS MODE" if locked else "PAGE NAVIGATION")
        self.mode_label.setProperty("locked", locked)
        self.mode_label.style().unpolish(self.mode_label)
        self.mode_label.style().polish(self.mode_label)

    def _on_item_selected(self, page_name: str, item_name: str) -> None:
        self.toast.setText(f"{page_name}: {item_name}")
        self.toast.show()
        QTimer.singleShot(1100, self.toast.hide)

    def _apply_style(self) -> None:
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(9, 12, 20))
        self.setPalette(palette)

        app_font = QFont("Segoe UI Variable")
        if not app_font.exactMatch():
            app_font = QFont("Segoe UI")
        QApplication.instance().setFont(app_font)

        self.setStyleSheet(
            """
            #appRoot {
                background:
                    qradialgradient(cx: 0.16, cy: 0.12, radius: 1.2,
                    fx: 0.16, fy: 0.12, stop: 0 rgba(34, 72, 96, 210),
                    stop: 0.44 rgba(12, 17, 28, 255),
                    stop: 1 rgba(7, 9, 16, 255));
            }

            #brand {
                color: rgba(244, 249, 255, 245);
                font-size: 28px;
                font-weight: 800;
            }

            #modePill {
                min-width: 190px;
                padding: 10px 18px;
                border-radius: 8px;
                color: rgba(226, 245, 255, 240);
                background: rgba(255, 255, 255, 28);
                border: 1px solid rgba(255, 255, 255, 50);
                font-size: 13px;
                font-weight: 800;
            }

            #modePill[locked="true"] {
                color: white;
                background: rgba(78, 220, 255, 90);
                border: 1px solid rgba(78, 220, 255, 210);
            }

            #pageIndicator {
                color: rgba(232, 242, 255, 220);
                font-size: 20px;
                font-weight: 700;
            }

            #pageViewport {
                background: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 44);
                border-radius: 8px;
            }

            #dashboardPage {
                background: transparent;
            }

            #pageTitle {
                color: white;
                font-size: 54px;
                font-weight: 850;
            }

            #pageSubtitle {
                color: rgba(218, 231, 244, 205);
                font-size: 20px;
                font-weight: 500;
            }

            #pageStatus {
                padding: 10px 16px;
                border-radius: 8px;
                color: rgba(222, 236, 248, 210);
                background: rgba(255, 255, 255, 24);
                border: 1px solid rgba(255, 255, 255, 38);
                font-size: 15px;
                font-weight: 700;
            }

            #pageStatus[locked="true"] {
                color: white;
                background: rgba(78, 220, 255, 70);
                border: 1px solid rgba(78, 220, 255, 180);
            }

            #focusTile {
                border-radius: 8px;
                background: rgba(18, 28, 42, 210);
                border: 1px solid rgba(255, 255, 255, 42);
            }

            #focusTile[focused="true"] {
                background: rgba(34, 69, 88, 235);
                border: 2px solid rgba(87, 230, 255, 245);
            }

            #tileIcon {
                color: white;
                font-size: 42px;
                font-weight: 800;
            }

            #tileTitle {
                color: white;
                font-size: 25px;
                font-weight: 800;
            }

            #tileSubtitle {
                color: rgba(222, 234, 246, 190);
                font-size: 15px;
                font-weight: 500;
            }

            #toast {
                margin: 0 340px;
                padding: 10px 18px;
                border-radius: 8px;
                color: white;
                background: rgba(78, 220, 255, 92);
                border: 1px solid rgba(120, 235, 255, 180);
                font-size: 17px;
                font-weight: 800;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = GestureDashboard()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
