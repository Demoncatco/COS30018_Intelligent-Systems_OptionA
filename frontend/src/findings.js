// Plain-language names for the 12 findings (keys match shared/schema.py FINDINGS exactly).

export const FINDING_INFO = {
  ACL: { label: "ACL injury", detail: "Anterior cruciate ligament, which stops the shin sliding forward" },
  MCL: { label: "MCL injury", detail: "Medial collateral ligament, on the inner side of the knee" },
  "Medial Meniscus": { label: "Medial meniscus tear", detail: "Cartilage cushion on the inner side of the joint" },
  "Lateral Meniscus": { label: "Lateral meniscus tear", detail: "Cartilage cushion on the outer side of the joint" },
  "Medial OA": { label: "Inner-joint osteoarthritis", detail: "Wear of the cartilage on the inner side" },
  "Lateral OA": { label: "Outer-joint osteoarthritis", detail: "Wear of the cartilage on the outer side" },
  "PF OA": { label: "Kneecap-joint osteoarthritis", detail: "Wear between the kneecap and thigh bone (patellofemoral)" },
  Effusion: { label: "Joint effusion", detail: "Extra fluid inside the knee joint" },
  Synovitis: { label: "Synovitis", detail: "Inflammation of the joint lining" },
  "Baker's": { label: "Baker's cyst", detail: "Fluid-filled swelling behind the knee" },
  Contusion: { label: "Bone bruise", detail: "Bruising inside the bone, often from impact or twisting" },
  Fracture: { label: "Fracture", detail: "A break in one of the knee bones" },
};

export const PLANE_LABEL = { sagittal: "Side view", coronal: "Front view", axial: "Top-down view" };
export const PLANE_SHORT = { sagittal: "S", coronal: "C", axial: "A" };
export const PLANES = ["sagittal", "coronal", "axial"];

export const URGENCY = {
  routine: { label: "Routine follow-up", tone: "calm" },
  see_doctor_soon: { label: "See a doctor soon", tone: "soon" },
  urgent: { label: "Seek care urgently", tone: "urgent" },
};

export const FLAG_TEXT = {
  low_slice_count: "One or more views had fewer than 15 slices, so some slices were repeated.",
};

export function describeFlag(flag) {
  if (flag.startsWith("missing_plane:")) {
    const plane = flag.split(":")[1];
    return `No ${PLANE_LABEL[plane]?.toLowerCase() ?? plane} (${plane}) series was found, so the result relies on the other views.`;
  }
  return FLAG_TEXT[flag] ?? flag;
}

export const pct = (p) => `${Math.round(p * 100)}%`;