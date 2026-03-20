// Stadium metadata with team colors for background theming.
// Each entry: token, name, team, primary (dark bg color), secondary (accent/gradient).

export const STADIUM_COLORS = {
  ATH: { primary: '#003831', secondary: '#EFB21E' },
  ATL: { primary: '#13274F', secondary: '#CE1141' },
  AZ:  { primary: '#1A1A1A', secondary: '#A71930' },
  BAL: { primary: '#1C1C1C', secondary: '#DF4601' },
  BOS: { primary: '#1A1A2E', secondary: '#BD3039' },
  CHC: { primary: '#0E3386', secondary: '#CC3433' },
  CIN: { primary: '#1A0A00', secondary: '#C6011F' },
  CLE: { primary: '#002B5C', secondary: '#E31937' },
  COL: { primary: '#1E0040', secondary: '#33006F' },
  CWS: { primary: '#1C1C1C', secondary: '#C4CED4' },
  DET: { primary: '#0C2C56', secondary: '#FA4616' },
  HOU: { primary: '#1A2035', secondary: '#EB6E1F' },
  KC:  { primary: '#174885', secondary: '#C09A5B' },
  LAA: { primary: '#1A0A12', secondary: '#BA0021' },
  LAD: { primary: '#001F5B', secondary: '#005A9C' },
  MIA: { primary: '#041E42', secondary: '#00A3E0' },
  MIL: { primary: '#12284B', secondary: '#FFC52F' },
  MIN: { primary: '#001B3D', secondary: '#D31145' },
  NYM: { primary: '#002D72', secondary: '#FF5910' },
  NYY: { primary: '#0D1B2E', secondary: '#003087' },
  PHI: { primary: '#1A0A12', secondary: '#E81828' },
  PIT: { primary: '#1C1C1C', secondary: '#FDB827' },
  SD:  { primary: '#1A1005', secondary: '#2F241D' },
  SEA: { primary: '#0C2C56', secondary: '#005C5C' },
  SF:  { primary: '#1A0A00', secondary: '#FD5A1E' },
  STL: { primary: '#1A0010', secondary: '#C41E3A' },
  TB:  { primary: '#092C5C', secondary: '#8FBCE6' },
  TEX: { primary: '#001639', secondary: '#C0111F' },
  TOR: { primary: '#1A2035', secondary: '#134A8E' },
  WSH: { primary: '#14001C', secondary: '#AB0003' },
}

export const OUTCOME_LABELS = {
  K:    'Strikeout',
  BIPO: 'Ball In Play Out',
  BB:   'Walk / HBP',
  '1B': 'Single',
  XBH:  'Extra-Base Hit',
  HR:   'Home Run',
}

export function getHeadshotUrl(mlbamId) {
  return `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${mlbamId}/headshot/67/current`
}
