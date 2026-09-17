#include "ImGuiUtils.h"

#include <imgui.h>

void labelThenRightAlignedWidget(const char* label, float widget_width) {
    ImGui::Spacing();
    float right_edge_x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", label);
    ImGui::SameLine();
    ImGui::SetCursorPosX(right_edge_x - widget_width);
    ImGui::SetNextItemWidth(widget_width);
}
