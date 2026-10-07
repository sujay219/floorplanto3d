export type ComponentKind =
  | 'base_cabinet'
  | 'drawer_cabinet'
  | 'wall_cabinet'
  | 'loft'
  | 'countertop'
  | 'sink'
  | 'hob'
  | 'tall_unit'
  | 'wardrobe';

export interface ComponentDefinition {
  kind: ComponentKind;
  label: string;
  description?: string;
  defaultDimensions: {
    width: number;
    depth: number;
    height: number;
  };
  tags?: string[];
}

export const componentCatalog: ComponentDefinition[] = [
  {
    kind: 'base_cabinet',
    label: 'Base Cabinet',
    defaultDimensions: { width: 600, depth: 600, height: 900 },
    tags: ['storage', 'base'],
  },
  {
    kind: 'drawer_cabinet',
    label: 'Drawer Cabinet',
    defaultDimensions: { width: 600, depth: 600, height: 900 },
    tags: ['storage', 'drawer'],
  },
  {
    kind: 'wall_cabinet',
    label: 'Wall Cabinet',
    defaultDimensions: { width: 600, depth: 350, height: 700 },
    tags: ['storage', 'wall'],
  },
  {
    kind: 'loft',
    label: 'Loft',
    defaultDimensions: { width: 1200, depth: 600, height: 200 },
    tags: ['storage', 'upper'],
  },
  {
    kind: 'countertop',
    label: 'Countertop',
    defaultDimensions: { width: 1200, depth: 600, height: 40 },
    tags: ['surface'],
  },
  {
    kind: 'sink',
    label: 'Sink',
    defaultDimensions: { width: 500, depth: 500, height: 200 },
    tags: ['plumbing'],
  },
  {
    kind: 'hob',
    label: 'Hob',
    defaultDimensions: { width: 600, depth: 500, height: 60 },
    tags: ['appliance'],
  },
  {
    kind: 'tall_unit',
    label: 'Tall Unit',
    defaultDimensions: { width: 600, depth: 600, height: 2100 },
    tags: ['storage', 'vertical'],
  },
  {
    kind: 'wardrobe',
    label: 'Wardrobe',
    defaultDimensions: { width: 1000, depth: 600, height: 2400 },
    tags: ['storage', 'wardrobe'],
  },
];

export const getComponentDefinition = (kind: ComponentKind): ComponentDefinition | undefined =>
  componentCatalog.find((component) => component.kind === kind);
