from __future__ import annotations

from typing import List

from .enum_mapping_agent import EnumSuggestion


def review_cli(suggestions: List[EnumSuggestion]) -> List[EnumSuggestion]:
    pending = [s for s in suggestions if s.status == "pending"]
    for idx, item in enumerate(pending, start=1):
        print(f"\n[{idx}/{len(pending)}] field={item.field_name} raw='{item.raw_value}' count={item.count}")
        print(f"suggested={item.suggested_value} confidence={item.confidence:.2f} reason={item.reasoning}")
        print(f"allowed={item.allowed_values}")
        while True:
            choice = input("[C]onfirm [R]emap [J]Reject [S]kip > ").strip().upper()
            if choice == "C":
                if item.suggested_value is None:
                    print("Cannot confirm: suggested_value is null")
                    continue
                note = input("note(optional): ").strip()
                item.confirm(note)
                break
            if choice == "R":
                remap = input("Select allowed value: ").strip()
                note = input("note(optional): ").strip()
                try:
                    item.remap(remap, note)
                except ValueError as exc:
                    print(str(exc))
                    continue
                break
            if choice == "J":
                note = input("note(optional): ").strip()
                item.reject(note)
                break
            if choice == "S":
                note = input("note(optional): ").strip()
                item.skip(note)
                break
    return suggestions


def review_streamlit(suggestions: List[EnumSuggestion]) -> List[EnumSuggestion]:
    import importlib.util

    if importlib.util.find_spec("streamlit") is None:
        return suggestions

    import streamlit as st  # type: ignore

    st.title("Enum Mapping Review")
    pending = [s for s in suggestions if s.status == "pending"]
    if not pending:
        st.success("No pending suggestions")
        return suggestions

    for item in pending:
        with st.expander(f"{item.field_name}: {item.raw_value}", expanded=True):
            st.write({"suggested": item.suggested_value, "confidence": item.confidence, "reasoning": item.reasoning})
            action = st.selectbox("Action", ["confirm", "remap", "reject", "skip"], key=f"a-{item.field_name}-{item.raw_value}")
            remap_val = st.selectbox("Remap value", [""] + item.allowed_values, key=f"r-{item.field_name}-{item.raw_value}")
            note = st.text_input("Note", key=f"n-{item.field_name}-{item.raw_value}")
            if st.button("Apply", key=f"b-{item.field_name}-{item.raw_value}"):
                try:
                    if action == "confirm":
                        item.confirm(note)
                    elif action == "remap":
                        item.remap(remap_val, note)
                    elif action == "reject":
                        item.reject(note)
                    else:
                        item.skip(note)
                except ValueError as exc:
                    st.error(str(exc))
                st.rerun()

    return suggestions
