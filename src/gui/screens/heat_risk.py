"""Heat Risk Dashboard Screen - What-if Analysis for Outdoor Activities"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QLabel, QPushButton, QComboBox, QScrollArea,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import requests
import json

from ..widgets.whatif_simulator import WhatIfSimulator
from ..widgets.action_card import ActionCardWidget


class HeatRiskScreen(QWidget):
    """Heat Risk Dashboard with What-If Analysis"""

    whatif_completed = pyqtSignal(dict)

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.current_location = "school_a"
        self.current_group = "elementary"
        self.hourly_data = []
        self.init_ui()

    def init_ui(self):
        """Initialize the heat risk dashboard UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 24, 24, 24)

        # Header
        self.create_header(main_layout)

        # Group selector
        self.create_group_selector(main_layout)

        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left: Hourly Risk Chart
        self.create_hourly_chart(content_layout)

        # Right: What-If Simulator
        self.whatif_simulator = WhatIfSimulator()
        self.whatif_simulator.simulate_clicked.connect(self.on_simulate_requested)
        content_layout.addWidget(self.whatif_simulator)

        main_layout.addLayout(content_layout)

        # Bottom: Action Card
        self.create_action_card_section(main_layout)

        # Load initial data
        self.load_risk_data()

    def create_header(self, parent_layout):
        """Create header with title and location selector"""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(2)

        title_label = QLabel("🌡️ Heat Risk Dashboard")
        title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 20px;
            font-weight: 600;
        """)
        title_layout.addWidget(title_label)

        subtitle_label = QLabel("วางแผนกิจกรรมกลางแจ้งอย่างปลอดภัย")
        subtitle_label.setStyleSheet("""
            color: #a0a0a0;
            font-size: 12px;
        """)
        title_layout.addWidget(subtitle_label)

        header_layout.addWidget(title_container)
        header_layout.addStretch()

        # Location selector
        location_container = QWidget()
        location_layout = QHBoxLayout(location_container)
        location_layout.setContentsMargins(0, 0, 0, 0)
        location_layout.setSpacing(8)

        location_label = QLabel("📍 พื้นที่:")
        location_label.setStyleSheet("color: #a0a0a0; font-size: 12px;")
        location_layout.addWidget(location_label)

        self.location_combo = QComboBox()
        self.location_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 150px;
            }
            QComboBox:hover { border-color: #4a4a4a; }
            QComboBox:focus { border-color: #0088ff; }
        """)
        self.location_combo.addItems([
            "โรงเรียน A",
            "โรงเรียน B",
            "โรงเรียน C",
            "สวนกลาง",
            "ตลาดสยาม"
        ])
        self.location_combo.currentIndexChanged.connect(self.on_location_changed)
        location_layout.addWidget(self.location_combo)

        header_layout.addWidget(location_container)
        parent_layout.addWidget(header_widget)

    def create_group_selector(self, parent_layout):
        """Create vulnerability group selection chips"""
        group_container = QWidget()
        group_layout = QHBoxLayout(group_container)
        group_layout.setSpacing(12)

        group_label = QLabel("👥 กลุ่มเปราะบาง:")
        group_label.setStyleSheet("color: #a0a0a0; font-size: 12px;")
        group_layout.addWidget(group_label)

        self.group_buttons = {}
        groups = [
            ("elementary", "👶 นักเรียนประถม", "#2196F3"),
            ("elderly", "👴 ผู้สูงอายุ", "#9C27B0"),
            ("riders", "🏍️ ไรเดอร์", "#FF5722"),
            ("workers", "👷 กรรมกร", "#607D8B"),
            ("general", "👥 ทั่วไป", "#4CAF50"),
        ]

        for group_id, label, color in groups:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(group_id == "elementary")

            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #2a2a2a;
                    color: #e0e0e0;
                    border: 1px solid #3a3a3a;
                    border-radius: 16px;
                    padding: 8px 16px;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: #3a3a3a;
                    border-color: {color};
                }}
                QPushButton:checked {{
                    background-color: {color};
                    color: #ffffff;
                    border-color: {color};
                }}
            """)
            btn.clicked.connect(lambda checked, g=group_id: self.on_group_selected(g))
            self.group_buttons[group_id] = btn
            group_layout.addWidget(btn)

        group_layout.addStretch()
        parent_layout.addWidget(group_container)

    def create_hourly_chart(self, parent_layout):
        """Create the hourly risk chart container"""
        chart_container = QFrame()
        chart_container.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
        """)
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setSpacing(12)
        chart_layout.setContentsMargins(16, 16, 16, 16)

        # Chart header
        chart_header = QWidget()
        header_layout = QHBoxLayout(chart_header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        chart_title = QLabel("⏰ ความเสี่ยงรายชั่วโมง")
        chart_title.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: 600;")
        header_layout.addWidget(chart_title)

        header_layout.addStretch()

        self.max_risk_label = QLabel("ความเสี่ยงสูงสุด: --")
        self.max_risk_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        header_layout.addWidget(self.max_risk_label)

        chart_layout.addWidget(chart_header)

        # Chart area with scroll
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)

        self.chart_content = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_content)
        self.chart_layout.setSpacing(8)

        self.chart_scroll.setWidget(self.chart_content)
        chart_layout.addWidget(self.chart_scroll, 1)

        parent_layout.addWidget(chart_container, 1)

    def create_action_card_section(self, parent_layout):
        """Create the Action Card section"""
        self.action_card = ActionCardWidget()
        parent_layout.addWidget(self.action_card)

    def load_risk_data(self):
        """Load risk data from API or generate sample data"""
        try:
            url = f"http://localhost:8000/api/risk/hourly"
            params = {
                "location_id": self.current_location,
                "group_id": self.current_group
            }
            response = requests.get(url, params=params, timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.hourly_data = data.get("hourly_risks", [])
            else:
                self.hourly_data = self.generate_sample_data()
        except Exception:
            self.hourly_data = self.generate_sample_data()

        self.update_chart()

    def generate_sample_data(self):
        """Generate sample hourly risk data"""
        base_temps = [
            28, 28, 27, 26, 26, 27,
            29, 31, 33, 35, 37, 38,
            39, 40, 39, 38, 36, 35,
            34, 33, 32, 31, 30, 29
        ]

        group_multipliers = {
            "elementary": 1.3,
            "elderly": 1.5,
            "riders": 1.2,
            "workers": 1.4,
            "general": 1.0
        }
        multiplier = group_multipliers.get(self.current_group, 1.0)

        data = []
        for hour, temp in enumerate(base_temps):
            if temp < 30:
                base_risk = (temp - 26) * 2
            elif temp < 35:
                base_risk = 8 + (temp - 30) * 4
            elif temp < 38:
                base_risk = 28 + (temp - 35) * 6
            else:
                base_risk = 46 + (temp - 38) * 8

            risk_score = min(100, base_risk * multiplier)

            level = "Low"
            if risk_score > 80:
                level = "Critical"
            elif risk_score > 60:
                level = "High"
            elif risk_score > 30:
                level = "Medium"

            data.append({
                "hour": hour,
                "temperature": temp,
                "risk_score": round(risk_score, 1),
                "level": level
            })

        return data

    def update_chart(self):
        """Update the hourly chart with current data"""
        while self.chart_layout.count():
            item = self.chart_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        hours_to_show = range(6, 21)  # 6 AM to 8 PM

        chart_grid = QWidget()
        grid_layout = QGridLayout(chart_grid)
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Header row
        grid_layout.addWidget(QLabel(""), 0, 0)
        for i, hour in enumerate(hours_to_show):
            label = QLabel(f"{hour:02d}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #606060; font-size: 10px;")
            grid_layout.addWidget(label, 0, i + 1)

        # Risk bar row
        grid_layout.addWidget(QLabel("Risk"), 1, 0)
        for i, hour in enumerate(hours_to_show):
            hour_data = next((h for h in self.hourly_data if h["hour"] == hour), None)
            if hour_data:
                risk = hour_data["risk_score"]
                level = hour_data["level"]

                if level == "Critical":
                    bar_color = "#F44336"
                elif level == "High":
                    bar_color = "#FF9800"
                elif level == "Medium":
                    bar_color = "#FFC107"
                else:
                    bar_color = "#4CAF50"

                bar = QFrame()
                bar.setFixedHeight(max(20, int(risk * 1.5)))
                bar.setStyleSheet(f"""
                    QFrame {{
                        background-color: {bar_color};
                        border-radius: 2px;
                    }}
                """)

                bar.setToolTip(
                    f"{hour:02d}:00\n"
                    f"อุณหภูมิ: {hour_data['temperature']}°C\n"
                    f"ความเสี่ยง: {risk:.0f} ({level})"
                )

                grid_layout.addWidget(bar, 1, i + 1)

        self.chart_layout.addWidget(chart_grid)

        if self.hourly_data:
            max_risk = max(self.hourly_data, key=lambda x: x["risk_score"])
            self.max_risk_label.setText(
                f"ความเสี่ยงสูงสุด: {max_risk['risk_score']:.0f} ({max_risk['level']}) "
                f"เวลา {max_risk['hour']:02d}:00"
            )
            self.whatif_simulator.set_hourly_data(self.hourly_data)

    def on_location_changed(self, index):
        """Handle location selection change"""
        locations = ["school_a", "school_b", "school_c", "park_central", "market_siam"]
        if index < len(locations):
            self.current_location = locations[index]
            self.load_risk_data()

    def on_group_selected(self, group_id):
        """Handle vulnerability group selection"""
        self.current_group = group_id

        for gid, btn in self.group_buttons.items():
            btn.setChecked(gid == group_id)

        self.load_risk_data()

    def on_simulate_requested(self, from_hour, to_hour):
        """Handle what-if simulation request"""
        from_data = next((h for h in self.hourly_data if h["hour"] == from_hour), None)
        to_data = next((h for h in self.hourly_data if h["hour"] == to_hour), None)

        if from_data and to_data:
            original_risk = from_data["risk_score"]
            recommended_risk = to_data["risk_score"]
            reduction = original_risk - recommended_risk
            reduction_pct = (reduction / original_risk * 100) if original_risk > 0 else 0

            result = {
                "original": {
                    "hour": from_hour,
                    "risk_score": original_risk,
                    "level": from_data["level"],
                    "temperature": from_data["temperature"]
                },
                "recommended": {
                    "hour": to_hour,
                    "risk_score": recommended_risk,
                    "level": to_data["level"],
                    "temperature": to_data["temperature"]
                },
                "improvement": {
                    "risk_reduction": reduction,
                    "level_change": f"{from_data['level']} → {to_data['level']}",
                    "percentage_reduction": f"{reduction_pct:.0f}%"
                }
            }

            self.whatif_completed.emit(result)
            self.update_action_card(result)

    def update_action_card(self, result):
        """Update the action card with what-if results"""
        recommendations = []

        if result.get("original", {}).get("level") == "Critical":
            recommendations.append({
                "title": "ยกเลิกกิจกรรมกลางแจ้ง",
                "icon": "🚫",
                "description": f"เลื่อนจากเวลา {result['original']['hour']:02d}:00 ไปเป็น {result['recommended']['hour']:02d}:00",
                "urgency": "critical"
            })

        if result.get("recommended", {}).get("level") in ["Low", "Medium"]:
            recommendations.append({
                "title": "ดำเนินกิจกรรมตามปกติ",
                "icon": "✅",
                "description": "ความเสี่ยงในระดับที่ยอมรับได้",
                "urgency": "low"
            })

        recommendations.extend([
            {
                "title": "เตรียมน้ำดื่ม",
                "icon": "💧",
                "description": "เตรียมน้ำดื่มเพียงพอสำหรับทุกคน",
                "urgency": "medium"
            },
            {
                "title": "พักในที่ร่ม",
                "icon": "🏠",
                "description": "จัดที่พักในร่มหรือเครื่องปรับอากาศ",
                "urgency": "medium"
            }
        ])

        self.action_card.set_recommendations(recommendations)

    def refresh_data(self):
        """Refresh risk data"""
        self.load_risk_data()