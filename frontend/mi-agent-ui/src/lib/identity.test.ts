import { describe, expect, it } from "vitest";
import {
  canSeeAdminControls,
  displayInitials,
  formatDisplayName,
  roleLabel,
  type UserIdentity,
} from "./identity";

describe("formatDisplayName (Entra display-name derivation)", () => {
  it("shortens a full name to first-initial + last name", () => {
    expect(formatDisplayName("Joshua Hall")).toBe("J. Hall");
    expect(formatDisplayName("Jane Smith")).toBe("J. Smith");
  });
  it("handles multi-part names by using first initial + final surname", () => {
    expect(formatDisplayName("Mary Jane Watson")).toBe("M. Watson");
  });
  it("falls back to the email when only an email claim is available", () => {
    expect(formatDisplayName("joshua.hall@example.com")).toBe("joshua.hall@example.com");
  });
  it("returns a single token unchanged and null when empty", () => {
    expect(formatDisplayName("Madonna")).toBe("Madonna");
    expect(formatDisplayName(null)).toBeNull();
    expect(formatDisplayName("")).toBeNull();
  });
});

describe("displayInitials", () => {
  it("uses first+last initials for a name and two chars for an email", () => {
    expect(displayInitials("Joshua Hall")).toBe("JH");
    expect(displayInitials("jane@example.com")).toBe("JA");
    expect(displayInitials(null)).toBe("–");
  });
});

describe("role-based control visibility", () => {
  const operator: UserIdentity = { authenticated: true, user: "Joshua Hall", isOperator: true, roles: ["operator"] };
  const clientUser: UserIdentity = { authenticated: true, user: "Jane Smith", isOperator: false, roles: ["client"] };

  it("shows admin controls for operators, hides them for clients and unknowns", () => {
    expect(canSeeAdminControls(operator)).toBe(true);
    expect(canSeeAdminControls(clientUser)).toBe(false);
    expect(canSeeAdminControls(null)).toBe(false); // fail-closed
  });
  it("labels the role", () => {
    expect(roleLabel(operator)).toBe("Operator");
    expect(roleLabel(clientUser)).toBe("Client");
    expect(roleLabel(null)).toBeNull();
  });
});
