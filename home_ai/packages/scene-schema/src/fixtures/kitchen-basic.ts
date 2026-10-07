import type { Scene } from '../index';

export const kitchenBasicFixture: Scene = {
  version: 1,
  units: 'mm',
  metadata: {
    name: 'Kitchen Basic Fixture',
    description: 'Reference kitchen scene used by layout and renderer integration tests.',
  },
  rooms: [
    {
      id: 'room_1',
      name: 'Kitchen',
      dimensions: { width: 4000, depth: 3000, height: 2800 },
      walls: [
        {
          id: 'wall_north',
          start: { x: 0, y: 0, z: 0 },
          end: { x: 4000, y: 0, z: 0 },
          thickness: 120,
          height: 2800,
        },
        {
          id: 'wall_east',
          start: { x: 4000, y: 0, z: 0 },
          end: { x: 4000, y: 0, z: 3000 },
          thickness: 120,
          height: 2800,
        },
        {
          id: 'wall_south',
          start: { x: 4000, y: 0, z: 3000 },
          end: { x: 0, y: 0, z: 3000 },
          thickness: 120,
          height: 2800,
        },
        {
          id: 'wall_west',
          start: { x: 0, y: 0, z: 3000 },
          end: { x: 0, y: 0, z: 0 },
          thickness: 120,
          height: 2800,
        },
      ],
      doors: [
        {
          id: 'door_01',
          start: { x: 3000, y: 0, z: 0 },
          end: { x: 3600, y: 0, z: 0 },
          width: 600,
          height: 2100,
        },
      ],
      windows: [
        {
          id: 'window_01',
          start: { x: 500, y: 0, z: 3000 },
          end: { x: 1800, y: 0, z: 3000 },
          width: 1300,
          height: 1200,
          sillHeight: 900,
        },
      ],
      objects: [
        {
          id: 'sink_01',
          type: 'sink',
          name: 'Sink',
          position: { x: 2500, y: 0, z: 500 },
          rotation: { x: 0, y: 0, z: 0 },
          dimensions: { width: 500, depth: 500, height: 200 },
          material: { name: 'stainless_steel', color: '#d1d5db' },
          anchor: { type: 'wall', targetId: 'wall_north', position: 'left' },
        },
        {
          id: 'cabinet_01',
          type: 'drawer_cabinet',
          name: 'Drawer cabinet',
          position: { x: 2500, y: 0, z: 1000 },
          rotation: { x: 0, y: 0, z: 0 },
          dimensions: { width: 600, depth: 600, height: 900 },
          material: { name: 'oak', color: '#b8925c' },
          anchor: { type: 'object', targetId: 'sink_01', position: 'right' },
        },
      ],
      metadata: { style: 'urban' },
    },
  ],
};

export default kitchenBasicFixture;
