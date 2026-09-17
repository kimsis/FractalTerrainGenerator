#pragma once

/*!
 *	Prints `label`, then positions the cursor so the widget that follows always ends flush with the
 *	right edge, `widget_width` wide, regardless of the label's own width — instead of its position (or,
 *	for non-input widgets like Button that ignore SetNextItemWidth, its right edge) drifting depending
 *	on how long each row's label happens to be.
 *	@param	label			The label text to print before the widget.
 *	@param	widget_width	How wide the following widget shall be. Pass a fixed slider width for
 *							sliders/inputs; for a Button, pass its own natural size
 *							(ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2.0f)
 *							so the button's actual right edge lands at the true right edge.
 */
void labelThenRightAlignedWidget(const char* label, float widget_width);
