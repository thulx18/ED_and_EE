from ED.infer import infer_single
from EE.test_one import infer

def event_extraction(data):
    '''
    data format: list
    ['TEXT1', 'TEXT2', ... ]
    '''
    ae_result = infer(data)
    results = []
    for data in ae_result:
        print(data)
        text = data['text']
        event_list, trigger_list = [], []
        for event in data['event_list']:
            trigger = event['trigger']
            trigger_start_index = event['trigger_start_index']
            trigger_list.append({'trigger': trigger, 'trigger_start_index': trigger_start_index})
            event_list.append({
                'event_type': '',
                'trigger': trigger,
                'arguments': event['arguments']
            })
        single_data = {'text': text, 'trigger_list': trigger_list}
        ed_result = infer_single(single_data, len(trigger_list))
        for i, _, event_type, _ in ed_result:
            event_list[i]['event_type'] = event_type
        results.append({'text': text, 'event_list': event_list})
    return results

if __name__ == '__main__':
    data = [{"text": "截至6月18日13时20分，四川省宜宾市长宁县6.0级地震已造成13人遇难。", "id": "", "event_list": []}]
    results = event_extraction(data)
    for result in results:
        print(result['text'])
        for i, event in enumerate(result['event_list']):
            event_type, trigger, arguments = event['event_type'], event['trigger'], event['arguments']
            print(f'# Event {i} || type: {event_type} , trigger: {trigger}')
            print(f'#   Argument:')
            for argument in arguments:
                role, argu = argument['role'], argument['argument']
                print(f'      {role} : {argu}')
        print()